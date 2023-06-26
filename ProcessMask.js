class Colors {
	constructor(
		canvas,
		lineWidth,
		lineColor,
		font,
		textColor,
		textBackgoundColor
	) {
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
	// self.pp = 'r'
	getColor(i) {
		const c = self.palette[i % self.n];
		return [c[0] / 255, c[1] / 255, c[2] / 255];
	}
}

const masks = (maskPatterns, colors, preprocImage, alpha) => {
	var colors = tf.cast(colors, 'float32').reshape([-1, 1, 1, 3]); //shape(n,1,1,3)
	// maskPatterns = tf.cast(maskPatterns, 'float32').expandDims(-1); // (n,h,w,1)

	const masksColor = maskPatterns.mul(colors.mul(alpha)); // shape(n,h,w,3)
	const invAlphMasks = tf.cumprod(tf.scalar(1).sub(maskPatterns.mul(alpha)), 0); // shape(n,h,w,1) where h=w=160

	const mcs = tf.sum(masksColor.mul(invAlphMasks), 0).mul(2); // mask color summand shape(n,h,w,3)
	const [invAlphMasksHead, invAlphMasksTail] = tf.split(
		invAlphMasks,
		[invAlphMasks.shape[0] - 1, 1],
		0
	);

	preprocImage = preprocImage.mul(invAlphMasksTail).add(mcs).mul(255);
	return preprocImage;
};

export { Colors, masks };
