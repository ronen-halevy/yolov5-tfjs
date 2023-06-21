const configRender = {
	font: '20px serif',
	lineWidth: 3,
	lineColor: 'yellow',
	textColor: 'blue',
	textBackgoundColor: 'white',
};
// tf.setBackend('webgl');

class YoloV5 {
	constructor(model, anchors, nClasses, scoreTHR, iouTHR, maxBoxes) {
		this.model = model;
		this.anchors = anchors;
		this.nClasses = nClasses;
		this.scoreTHR = scoreTHR;
		this.iouTHR = iouTHR;
		this.maxBoxes = maxBoxes;
	}

	setScoreTHR = (val) => {
		this.scoreTHR = val;
	};
	setIouTHR = (val) => {
		this.iouTHR = val;
	};

	setMaxBoxes = (val) => {
		this.maxBoxes = val;
	};
	setModelParams(model, anchors, nClasses) {
		this.model = model;
		this.anchors = anchors;
		this.nClasses = nClasses;
	}

	imagePreprocess = (image) => {
		const imgTensor = tf.browser.fromPixels(image);
		// Yolo input width:
		const imageHeight = 640;
		const imageWidth = 640;
		var resized = tf.image.resizeBilinear(imgTensor, [imageHeight, imageWidth]);
		var tensor = resized.expandDims(0).toFloat();
		tensor = tensor.div(255);
		return tensor;
	};

	xywh2xyxy = (x) => {
		// Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
		const axis = -1;
		let [xc, yc, w, h] = tf.split(x, [1, 1, 1, 1], axis);

		const xmin = xc.sub(w.div(2)); // top left x
		const ymin = yc.sub(h.div(2)); // top left y
		const xmax = xc.add(w.div(2)); // bottom right x
		const ymax = yc.add(h.div(2)); // bottom right y
		return tf.concat([xmin, ymin, xmax, ymax], axis);
	};

	detectFrame = (imageFrame) => {
		// tf.engine cleans intermidiate allocations avoids memory leak - equivalent to tf.tidy
		tf.engine().startScope();

		const imageTensor = this.imagePreprocess(imageFrame);
		const [proto, modelOutputGrids] = this.model.predict(imageTensor);
		const nm = 32;

		const pred = modelOutputGrids.squeeze(0);
		const nc = pred.shape[1] - nm - 5; // n of classes
		const mi = 5 + nc; // mask start index

		let axis = -1;
		let [bboxes, confidences, classProbs, masks] = tf.split(
			pred,
			[4, 1, nc, nm],
			axis
		);
		let classIndices = classProbs.argMax(axis);
		classProbs = classProbs.max(axis);
		confidences = confidences.squeeze(axis);
		let scores = confidences.mul(classProbs);
		classProbs.dispose();
		confidences.dispose();
		bboxes = this.xywh2xyxy(bboxes);

		const nmsPromise = nms(
			bboxes,
			scores,
			classIndices,
			masks,
			this.iouTHR,
			this.scoreTHR,
			this.maxBoxes
		).then((reasultArrays) => {
			tf.engine().endScope();

			return reasultArrays;
		});
		return nmsPromise;
	};
}

const nms = (
	bboxes,
	scores,
	classIndices,
	masks,
	iouTHR,
	scoreTHR,
	maxBoxes
) => {
	const nmsPromise = new Promise((resolve) => {
		const nmsResults = tf.image.nonMaxSuppressionAsync(
			bboxes,
			scores,
			maxBoxes,
			iouTHR,
			scoreTHR
		);
		resolve(nmsResults);
	}).then((nmsResults) => {
		let selectedBboxes = bboxes.gather(nmsResults);
		let selectedClasses = classIndices.gather(nmsResults);
		let selectedScores = scores.gather(nmsResults);
		let selectedMasks = masks.gather(nmsResults);

		const bboxesArray = selectedBboxes.array();
		const scoresArray = selectedScores.array();
		const classIndicesArray = selectedClasses.array();
		const masksArray = selectedMasks.array();

		let reasultArrays = Promise.all([
			bboxesArray,
			scoresArray,
			classIndicesArray,
			masksArray,
		]);

		selectedBboxes.dispose();
		selectedClasses.dispose();
		selectedScores.dispose();
		selectedMasks.dispose();
		return reasultArrays;
	});

	return nmsPromise;
};

function arrange_bbox(xy, wh) {
	let grid_size = [xy.shape[1], xy.shape[1]];

	let grid = tf.meshgrid(
		tf.range(0, xy.shape[1], 1),
		tf.range(0, xy.shape[1], 1)
	);
	var axis = -1;
	grid = tf.stack(grid, axis);

	axis = 2;
	grid = grid.expandDims(axis);

	xy = xy.add(tf.cast(grid, 'float32'));
	xy = xy.div(tf.cast(grid_size, 'float32'));

	let value1 = tf.scalar(2);
	wh = wh.div(value1);
	var xy_min = xy.sub(wh);
	var xy_max = xy.add(wh);

	var bbox = tf.concat([xy_min, xy_max], -1);
	grid.dispose();
	grid.dispose();

	return bbox;
}

const createModel = (modelUrl, anchorsUrl, classNamesUrl) => {
	const modelPromise = tf.loadGraphModel(modelUrl);
	const anchorsPromise = fetch(anchorsUrl).then((response) => response.json());
	const classNamesPromise = fetch(classNamesUrl).then((x) => x.text());

	const promise = Promise.all([
		modelPromise,
		anchorsPromise,
		classNamesPromise,
	]);
	return promise;
};

// export { YoloV3 };
// const yolo = {
//   YoloPredictor: YoloPredictor,
// };

// module.exports = yolo;
export { YoloV5, createModel };
