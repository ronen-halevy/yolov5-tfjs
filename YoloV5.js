import { Colors, masks } from './ProcessMask.js';

// crop masks pixels which are not inside a bbox.
const cropMask = (masks, boxes) => {
	const [n, h, w] = masks.shape;
	const [xmin, ymin, xmax, ymax] = tf.split(
		boxes.expandDims([1]),
		[1, 1, 1, 1],
		-1
	);

	const r = tf.range(0, w, 1, xmin.dtype).expandDims(0).expandDims(0); // array [0.....,w-1] dim: [1,1,w]
	const c = tf.range(0, h, 1, xmin.dtype).expandDims(-1).expandDims(0); // array [0.....,h-1] dim: [1,h,1]

	// crop is masks pixels which are not inside a bbox.

	const crop = r
		.greaterEqual(xmin)
		.mul(r.lessEqual(xmax))
		.mul(c.greaterEqual(ymin))
		.mul(c.lessEqual(ymax));

	return masks.mul(crop); // ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))
};

const processMask = (protos, masksIn, bboxes, ih, iw) => {
	var [ch, mh, mw] = protos.shape;
	const protosCols = protos.reshape([ch, -1]);
	var masks = masksIn.matMul(protosCols).sigmoid().reshape([-1, mh, mw]);

	var downsampled_bboxes = $.extend({}, bboxes);

	var [xmin, ymin, xmax, ymax] = tf.split(downsampled_bboxes, [1, 1, 1, 1], -1);
	downsampled_bboxes = tf.concat(
		[
			xmin.mul(masks.shape[1]),
			ymin.mul(masks.shape[2]),
			xmax.mul(masks.shape[1]),
			ymax.mul(masks.shape[2]),
		],
		-1
	);
	return cropMask(masks, downsampled_bboxes);
};

class YoloV5 {
	constructor(model, nClasses, scoreTHR, iouTHR, maxBoxes) {
		this.model = model;
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

	setModelParams = (model, nClasses) => {
		this.model = model;
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

	detectFrame = async (imageFrame) => {
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
		// const mi = 5 + nc; // mask start index

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

		// ronen - var only bbox todo
		var [selBboxes, selScores, selclassIndices, selMasksCoeffs] = await nms(
			bboxes,
			scores,
			classIndices,
			masksCoeffs,
			this.iouTHR,
			this.scoreTHR,
			this.maxBoxes
		);

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

		const bboxesArray = selBboxes.array();
		const scoresArray = selScores.array();
		const classIndicesArray = selclassIndices.array();
		const masksResArray = masksRes.array();

		// scaledBoxes.dispose();
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

const createModel = (modelUrl, classNamesUrl) => {
	const modelPromise = tf.loadGraphModel(modelUrl);
	const classNamesPromise = fetch(classNamesUrl).then((x) => x.text());

	const promise = Promise.all([modelPromise, classNamesPromise]);
	return promise;
};

export { YoloV5, createModel };
