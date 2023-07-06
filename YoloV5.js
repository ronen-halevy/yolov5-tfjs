class ProcMasks {
	palette = tf.tensor([
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
	]);

	/// crop out masks which are not bbox bounded. (note: )
	// inputs:
	// masks: dim [n,160,160],  True for max pixels False otherwise
	// boxes: dim [n, 4]
	cropMask = (masks, bboxes) => {
		return tf.tidy(() => {
			const [n, h, w] = masks.shape;

			// resize boxes to masks' dims i.e. 160*160 (modelSize/4):
			const downsampledBboxes = bboxes.mul([
				masks.shape[1],
				masks.shape[2],
				masks.shape[1],
				masks.shape[2],
			]);

			const [xmin, ymin, xmax, ymax] = tf.split(
				downsampledBboxes.expandDims([1]),
				[1, 1, 1, 1],
				-1
			);

			const r = tf.range(0, w, 1, xmin.dtype).expandDims(0).expandDims(0); // array [0.....,w-1] dim: [1,1,w]
			const c = tf.range(0, h, 1, xmin.dtype).expandDims(-1).expandDims(0); // array [0.....,h-1] dim: [1,h,1]

			// crop is True for masks pixels which are not inside a bbox.
			const crop = r
				.greaterEqual(xmin)
				.mul(r.lessEqual(xmax))
				.mul(c.greaterEqual(ymin))
				.mul(c.lessEqual(ymax));

			return masks.mul(crop); // ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))
		});
	};

	// input:
	// protos: 32, 160, 160
	// masksIn: n, 32
	// bboxes: n,4
	composeMasks = (protos, masksIn, bboxes) => {
		var [ch, mh, mw] = protos.shape;
		const protosCols = protos.reshape([ch, -1]);
		const masks = masksIn.matMul(protosCols).sigmoid().reshape([-1, mh, mw]);

		return this.cropMask(masks, bboxes);
	};

	// Composes an image with a mask overlay. The input image
	//inputs:
	// maskPatterns: a mask per detection, mask bits are True, type: bool  dim [n, h,w,1]
	// colors:  RGB bytes color per each detection type: range: [0,1] (pixels herer are floats),  float32 dim [n,3]
	// preprocImage: Input image, resized to model's dims
	// alpha: mask opacity float, scalar range [0.5,1].
	composeImage = (maskPatterns, colors, preprocImage, alpha) => {
		return tf.tidy(() => {
			colors = colors.expandDims(-2).expandDims(-2); //shape(n,1,1,3)
			maskPatterns = tf.cast(maskPatterns, 'float32'); // (n,h,w,1)

			const coloredMasks = maskPatterns.mul(colors.mul(alpha)); // shape(n,h,w,3)
			// invAlphMasks=cumprod(1-maskPatterns*alpha), where:
			//cumprod:  cumulative product of (1-maskPatterns*alpah), i.e.:
			// [(1-maskPatterns[0]*alpah), (1-maskPatterns[0]*alpah)(1-maskPatterns[1]*alpah)....(1-maskPatterns*alpah[0])*..*(1-maskPatterns*alpah[n])]
			// So in the nth element, invAlphMasks[n,i,j,1] is (1-alpha)^m, where m is the mumber of overlapping detections on this pixel.

			// invAlphMasks<0.5 if alpha>0.5. This confirms image+mask < 0.5
			const invAlphMasks = tf.cumprod(
				tf.scalar(1).sub(maskPatterns.mul(alpha)),
				0
			); // shape(n,h,w,1) where h=w=160

			// the colored masks are multiplied by invAlphMasks. Larger invAlphMasks is less opacity.
			// mul(2) brightens mask. mcs*2 < 0.5 as coloredMasks<0.5 and invAlphMasks<0.5^m, where m is num of detections on pixel
			const mcs = tf.sum(coloredMasks.mul(invAlphMasks), 0).mul(2); // mask color summand shape(n,h,w,3)

			const [invAlphMasksHead, invAlphMasksTail] = tf.split(
				invAlphMasks,
				[invAlphMasks.shape[0] - 1, 1],
				0
			);
			invAlphMasksHead.dispose();
			// preprocImage*invAlphMasksTail+: mcs (mask color summand)
			// Mul preprocImage by invAlphMasksTail confirms value <0., otherwise the sum might exceed 1.0
			const composedImage = preprocImage
				.squeeze(0)
				.mul(invAlphMasksTail.squeeze(0))
				.add(mcs);

			return [composedImage, mcs];
		});
	};

	run = (preprocImage, protos, selMasksCoeffs, selBboxes, selclassIndices) => {
		return tf.tidy(() => {
			var maskPatterns = this.composeMasks(
				protos.squeeze(0),
				selMasksCoeffs,
				selBboxes
			);
			protos.dispose();
			selMasksCoeffs.dispose();

			maskPatterns = tf.image
				.resizeBilinear(maskPatterns.expandDims(-1), [640, 640])
				.greater(0.5);

			const ind = selclassIndices.mod(this.palette.shape[0]);
			const colorPalette = this.palette.gather(tf.cast(ind, 'int32')).div(255);
			ind.dispose();
			// ronen - tbd todo add to config param:
			const alpha = 0.5;
			const [composedImage, masks] = this.composeImage(
				maskPatterns,
				colorPalette,
				preprocImage,
				alpha
			);
			colorPalette.dispose();

			return [composedImage, masks];
		});
	};
}

class YoloV5 {
	constructor(model, nClasses, scoreTHR, iouTHR, maxBoxes) {
		this.model = model;
		this.nClasses = nClasses;
		this.scoreTHR = scoreTHR;
		this.iouTHR = iouTHR;
		this.maxBoxes = maxBoxes;
		this.procMasks = new ProcMasks();
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

	setModelParams = (model, nClasses) => {
		this.model = model;
		this.nClasses = nClasses;
	};
	xywh2xyxy = (x) => {
		// Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
		return tf.tidy(() => {
			var [xy, wh] = tf.split(x, [2, 2], -1);
			var xyMin = xy.sub(wh.div(2));
			var xyMax = xy.add(wh.div(2));
			return tf.concat([xyMin, xyMax], -1);
		});
	};
	predict = (preprocImage) => {
		return tf.tidy(() => {
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
			preds.dispose();

			var classIndices = classProbs.argMax(axis);
			classProbs = classProbs.max(axis);
			confidences = confidences.squeeze(axis);
			const scores = confidences.mul(classProbs);
			bboxes = this.xywh2xyxy(bboxes);

			classProbs.dispose();
			confidences.dispose();
			return [protos, bboxes, classIndices, scores, masksCoeffs];
		});
	};

	detectFrame = async (imageFrame) => {
		// tf.engine cleans intermidiate allocations avoids memory leak - equivalent to tf.tidy
		tf.engine().startScope();

		const imageHeight = 640;
		const imageWidth = 640;

		const preprocImage = tf.tidy(() => {
			return tf.image
				.resizeBilinear(tf.browser.fromPixels(imageFrame), [
					imageHeight,
					imageWidth,
				])
				.div(255.0)
				.expandDims(0);
		});

		const [protos, bboxes, classIndices, scores, masksCoeffs] =
			this.predict(preprocImage);

		const [selBboxes, selScores, selclassIndices, selMasksCoeffs] = await nms(
			bboxes,
			scores,
			classIndices,
			masksCoeffs,
			this.iouTHR,
			this.scoreTHR,
			this.maxBoxes
		);
		scores.dispose();

		// bypass masks if no detections:
		let [composedImage, masks] = [null, null];
		if (selBboxes.shape[0] != 0) {
			[composedImage, masks] = this.procMasks.run(
				preprocImage,
				protos,
				selMasksCoeffs,
				selBboxes,
				selclassIndices
			);
		} else {
			composedImage = preprocImage.squeeze(0);
			masks = tf.zeros(composedImage.shape);
		}

		const bboxesArray = selBboxes.array();
		const scoresArray = selScores.array();
		const classIndicesArray = selclassIndices.array();
		const composedImageArray = composedImage.array();
		const masksArray = masks.array();

		tf.engine().endScope();

		// conversion to array is asunc:
		return Promise.all([
			bboxesArray,
			scoresArray,
			classIndicesArray,
			composedImageArray,
			masksArray,
		]);
	};
}

const nms = async (
	bboxes,
	scores,
	classIndices,
	masks,
	iouTHR,
	scoreTHR,
	maxBoxes
) => {
	const nmsResults = await tf.image.nonMaxSuppressionAsync(
		bboxes,
		scores,
		maxBoxes,
		iouTHR,
		scoreTHR
	);
	return [
		bboxes.gather(nmsResults),
		scores.gather(nmsResults),
		classIndices.gather(nmsResults),
		masks.gather(nmsResults),
	];
};

const createModel = async (modelUrl, classNamesUrl, onProgress) => {
	// in case of local url, eval to have the back ticked (`) address:
	if (modelUrl.includes('window.location.href')) {
		modelUrl = eval(modelUrl);
	}
	const model = await tf.loadGraphModel(modelUrl, {
		onProgress: onProgress,
	});
	const classNames = await fetch(classNamesUrl).then((x) => x.text());
	return [model, classNames];
};

export { YoloV5, createModel };
