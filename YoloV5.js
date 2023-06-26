import { Colors, masks } from './ProcessMask.js';

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

const processMask = (protos, masksIn, bboxes, ih, iw) => {
	var [ch, mh, mw] = protos.shape;
	// masks =  tf.sigmoid(masks_in @ tf.reshape(protos, [ch, -1])  )# CHW
	const protosCols = protos.reshape([ch, -1]); //.print();
	var masks = masksIn.matMul(protosCols).sigmoid().reshape([-1, mh, mw]);

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
	masks.masks = crop_mask(masks, downsampled_bboxes);
	// masks = tf.image.resizeBilinear(masks.expandDims(-1), [640, 640]);

	return masks;
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
		this.colors = new Colors();
		self.palette = [
			[0xff, 0x38, 0x38],
			[0xff, 0x9d, 0x97],
			[0xff, 0x70, 0x1f],
			[0xff, 0xb2, 0x1d],
			[0xcf, 0xd2, 0x31],
			[0x48, 0xf9, 0x0a],
			[0x92, 0xcc, 0x17],
			[0x3d, 0xdb, 0x86],
			[0x1a, 0x93, 0x34],
			[0x00, 0xd4, 0xbb],
			[0x2c, 0x99, 0xa8],
			[0x00, 0xc2, 0xff],
			[0x34, 0x45, 0x93],
			[0x64, 0x73, 0xff],
			[0x00, 0x18, 0xec],
			[0x84, 0x38, 0xff],
			[0x52, 0x00, 0x85],
			[0xcb, 0x38, 0xff],
			[0xff, 0x95, 0xc8],
			[0xff, 0x37, 0xc7],
		];
		self.n = self.palette.length;
	}
	getColor = (i) => {
		const c = self.palette[i % self.n];
		return [c[0] / 255, c[1] / 255, c[2] / 255];
	};
	setScoreTHR = (val) => {
		this.scoreTHR = val;
	};
	setIouTHR = (val) => {
		this.iouTHR = val;
	};

	setMaxBoxes = (val) => {
		this.maxBoxes = val;
	};

	setModelParams = (model, anchors, nClasses) => {
		this.model = model;
		this.anchors = anchors;
		this.nClasses = nClasses;
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

	detectFrame = async (imageFrame, canvas) => {
		// tf.engine cleans intermidiate allocations avoids memory leak - equivalent to tf.tidy
		tf.engine().startScope();
		const imageHeight = 640;
		const imageWidth = 640;
		const imageTensor = tf.browser.fromPixels(imageFrame);
		const preprocImage = tf.image
			.resizeBilinear(imageTensor, [imageHeight, imageWidth])
			.div(255.0)
			.expandDims(0);

		var [protos, preds] = this.model.predict(preprocImage);
		const nm = 32;

		preds = preds.squeeze(0);
		const nc = preds.shape[1] - nm - 5; // n of classes
		const mi = 5 + nc; // mask start index

		var axis = -1;
		var [bboxes, confidences, classProbs, masksCoeffs] = tf.split(
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

		const [selBboxes, selScores, selclassIndices, selMasksCoeffs] = await nms(
			bboxes,
			scores,
			classIndices,
			masksCoeffs,
			this.iouTHR,
			this.scoreTHR,
			this.maxBoxes
		);
		// return [selBboxes, selScores, selclassIndices, selMasks];
		protos = protos.squeeze(0);
		var maskPatterns = processMask(
			protos,
			selMasksCoeffs,
			selBboxes,
			imageFrame.width,
			imageFrame.height
		);
		maskPatterns = tf.image
			.resizeBilinear(maskPatterns.expandDims(-1), [640, 640])
			.greater(0.5);

		const selclassIndicesArr = await selclassIndices.array();
		const colorPalette = selclassIndicesArr.map((selclassIndex) =>
			this.getColor(selclassIndex)
		);
		const alpha = 0.5;
		const colorPaletteTens = tf.cast(colorPalette, 'float32');
		const masksRes = masks(maskPatterns, colorPaletteTens, preprocImage, alpha);
		await tf.browser.toPixels(masksRes, canvas);

		// tf.data.array(selclassIndices).forEachAsync((e) => console.log(e));

		const scaledBoxes = scale_boxes(
			preprocImage.shape.slice(1, 3),
			selBboxes,
			imageFrame.width,
			imageFrame.height
		); //.round(); // rescale boxes to im0 size

		const bboxesArray = selBboxes.array();
		const scoresArray = selScores.array();
		const classIndicesArray = selclassIndices.array();
		const masksResArray = masksRes.array();

		scaledBoxes.dispose();
		selBboxes.dispose();
		selScores.dispose();
		selclassIndices.dispose();
		selMasksCoeffs.dispose();
		// masksRes.dispose();
		tf.engine().endScope();

		var reasultArrays = Promise.all([
			bboxesArray,
			scoresArray,
			classIndicesArray,
			masksResArray,
		]);

		return reasultArrays;
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
