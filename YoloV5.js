// scale img1 scaled boxes to img0 scale: sub padded value and rescale
const scale_boxes = (img1_shape, boxes, img0_w, img0_h) => {
	// Rescale boxes (xyxy) from img1_shape to img0_shape
	// gain - resize factor
	const gain = Math.min(img1_shape[0] / img0_w, img1_shape[1] / img0_h); // gain  = old / new
	//
	const pad = tf.tensor1d([
		(img1_shape[1] - img0_h * gain) / 2,
		(img1_shape[0] - img0_w * gain) / 2,
	]); // wh padding
	// calc (boxes-pad)/gain:
	const tiledPad = pad.tile([2]).expandDims(0).tile([boxes.shape[0], 1]);
	boxes = boxes.sub(tiledPad).div(gain);

	// clip to image dims:
	var [xmin, ymin, xmax, ymax] = tf.split(boxes, [1, 1, 1, 1], -1);
	xmin = xmin.clipByValue(0, img0_w);
	ymin = xmax.clipByValue(0, img0_w);
	xmax = ymin.clipByValue(0, img0_h);
	ymax = ymax.clipByValue(0, img0_h);
	boxes = tf.concat([xmin, ymin, xmax, ymax], -1);

	return boxes;
};

const crop_mask = (masks, boxes) => {
	const [n, h, w] = masks.shape;
	const [xmin, ymin, xmax, ymax] = tf.split(
		boxes.expandDims([1]),
		[1, 1, 1, 1],
		-1
	);

	const r = tf.range(0, w, 1, xmin.dtype).expandDims(0).expandDims(0);
	const c = tf.range(0, h, 1, xmin.dtype).expandDims(-1).expandDims(0);
	const crop =
		tf.cast(r >= xmin, 'float32') *
		tf.cast(r < xmax, 'float32') *
		tf.cast(c >= ymin, 'float32') *
		tf.cast(c < ymax, 'float32');
	return masks.mul(crop); // ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))
};

const process_mask = (protos, masksIn, bboxes, ih, iw) => {
	var [ch, mh, mw] = protos.shape;
	// masks =  tf.sigmoid(masks_in @ tf.reshape(protos, [ch, -1])  )# CHW
	const protosCols = protos.reshape([ch, -1]); //.print();
	var masks = masksIn.matMul(protosCols).sigmoid();
	const masks1 = masks.reshape([-1, mh, mw]);

	var downsampled_bboxes = $.extend({}, bboxes);
	// console.log('ggggg');
	// /downsampled_bboxes = tf.concat([mw / iw, mh / ih, mw / iw, mh / ih], -1);
	var [xmin, ymin, xmax, ymax] = tf.split(downsampled_bboxes, [1, 1, 1, 1], -1);
	downsampled_bboxes = tf.concat(
		[
			xmin.mul(mw / iw),
			ymin.mul(mh / ih),
			xmax.mul(mw / iw),
			ymax.mul(mh / ih),
		],
		-1
	);
	masks = crop_mask(masks, downsampled_bboxes);
	return masks.greater(0.5);

	// downsampled_bboxes[:, 0] *= mw / iw
	// downsampled_bboxes[:, 2] *= mw / iw
	// downsampled_bboxes[:, 3] *= mh / ih
	// downsampled_bboxes[:, 1] *= mh / ih

	// ih, (iw = shape);
	// # masks = (masks_in @ tf.Variable(protos.reshape(ch, -1))).sigmoid().reshape(-1, mh, mw)  # CHW
	// masks =  tf.sigmoid(masks_in @ tf.reshape(protos, [ch, -1])  )# CHW
	// masks = tf.reshape(masks, (-1, mh, mw))
	// downsampled_bboxes = bboxes.copy()
	// downsampled_bboxes[:, 0] *= mw / iw
	// downsampled_bboxes[:, 2] *= mw / iw
	// downsampled_bboxes[:, 3] *= mh / ih
	// downsampled_bboxes[:, 1] *= mh / ih
	// masks = crop_mask(masks, downsampled_bboxes)  # CHW
	// if upsample:
	//     # masks = F.interpolate(masks[None], shape, mode='bilinear', align_corners=False)[0]  # CHW
	//     masks = tf.image.resize(masks[...,tf.newaxis], size=shape )
	// return tf.math.greater(
	//     tf.squeeze(masks, axis=-1), 0.5, name=None
	// )
	// # return masks.gt_(0.5)
};

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
		var [xc, yc, w, h] = tf.split(x, [1, 1, 1, 1], axis);

		const xmin = xc.sub(w.div(2)); // top left x
		const ymin = yc.sub(h.div(2)); // top left y
		const xmax = xc.add(w.div(2)); // bottom right x
		const ymax = yc.add(h.div(2)); // bottom right y
		return tf.concat([xmin, ymin, xmax, ymax], axis);
	};

	detectFrame = async (imageFrame) => {
		// tf.engine cleans intermidiate allocations avoids memory leak - equivalent to tf.tidy
		tf.engine().startScope();

		const imageTensor = this.imagePreprocess(imageFrame);
		var [protos, preds] = this.model.predict(imageTensor);
		const nm = 32;

		preds = preds.squeeze(0);
		const nc = preds.shape[1] - nm - 5; // n of classes
		const mi = 5 + nc; // mask start index

		var axis = -1;
		var [bboxes, confidences, classProbs, masks] = tf.split(
			preds,
			[4, 1, nc, nm],
			axis
		);
		var classIndices = classProbs.argMax(axis);
		classProbs = classProbs.max(axis);
		confidences = confidences.squeeze(axis);
		var scores = confidences.mul(classProbs);
		classProbs.dispose();
		confidences.dispose();
		bboxes = this.xywh2xyxy(bboxes);

		const [selBboxes, selScores, selclassIndices, selMasks] = await nms(
			bboxes,
			scores,
			classIndices,
			masks,
			this.iouTHR,
			this.scoreTHR,
			this.maxBoxes
		);
		// return [selBboxes, selScores, selclassIndices, selMasks];
		protos = protos.squeeze(0);
		masks = process_mask(
			protos,
			selMasks,
			selBboxes,
			imageFrame.width,
			imageFrame.height
		);

		const scaledBoxes = scale_boxes(
			imageTensor.shape.slice(1, 3),
			selBboxes,
			imageFrame.width,
			imageFrame.height
		).round(); // rescale boxes to im0 size

		const bboxesArray = selBboxes.array();
		const scoresArray = selScores.array();
		const classIndicesArray = selclassIndices.array();
		const masksArray = selMasks.array();

		scaledBoxes.dispose();
		selBboxes.dispose();
		selScores.dispose();
		selclassIndices.dispose();
		selMasks.dispose();

		tf.engine().endScope();
		var reasultArrays = Promise.all([
			bboxesArray,
			scoresArray,
			classIndicesArray,
			masksArray,
		]);
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
		var selectedBboxes = bboxes.gather(nmsResults);
		var selectedClasses = classIndices.gather(nmsResults);
		var selectedScores = scores.gather(nmsResults);
		var selectedMasks = masks.gather(nmsResults);

		var reasultArrays = Promise.all([
			selectedBboxes,
			selectedScores,
			selectedClasses,
			selectedMasks,
		]);

		return reasultArrays;
	});

	return nmsPromise;
};

function arrange_bbox(xy, wh) {
	var grid_size = [xy.shape[1], xy.shape[1]];

	var grid = tf.meshgrid(
		tf.range(0, xy.shape[1], 1),
		tf.range(0, xy.shape[1], 1)
	);
	var axis = -1;
	grid = tf.stack(grid, axis);

	axis = 2;
	grid = grid.expandDims(axis);

	xy = xy.add(tf.cast(grid, 'float32'));
	xy = xy.div(tf.cast(grid_size, 'float32'));

	var value1 = tf.scalar(2);
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

export { YoloV5, createModel };
