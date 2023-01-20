import { YoloV3, createModel } from './YoloV3.js';
import Render from './Render.js';

import configModel from './configs/configModel.json' assert { type: 'json' };
import configNms from './configs/configNms.json' assert { type: 'json' };
import configRender from './configs/configRender.json' assert { type: 'json' };

const modelsTable = configModel.models;
var classNames = '';
var yoloV3 = '';

const imageUrl =
	'https://cdn.pixabay.com/photo/2023/01/01/16/35/street-7690347_960_720.jpg';

const onSelectModel = async (event) => {
	$('#runYolo').attr('disabled', true);

	$('#loadingModelSpinner').show();

	console.log('onSelectModel', event.target.value);
	console.log('onSelectModel', event.target);
	const { modelUrl, anchorsUrl, classNamesUrl } =
		modelsTable[event.target.value];

	const [model, anchors, classNamesString] = await createModel(
		modelUrl,
		anchorsUrl,
		classNamesUrl
	);
	classNames = classNamesString.split(/\r?\n/);
	const nClasses = classNames.length;
	const { scoreTHR, iouTHR, maxBoxes } = configNms;
	yoloV3 = new YoloV3(
		model,
		anchors.anchor,
		nClasses,
		scoreTHR,
		iouTHR,
		maxBoxes
	);

	$('#loadingModelSpinner').hide();
	$('#runYolo').attr('disabled', false);
};

const onClickRunYolo = async (yoloV3, draw, imageUrl) => {
	$('#runYoloSpinner').show();
	var imageObject = new window.Image();
	const res = await fetch(imageUrl);
	const imageBlob = await res.blob();
	const imageObjectURL = URL.createObjectURL(imageBlob);
	imageObject.src = imageObjectURL;
	imageObject.addEventListener('load', async () => {
		const [selBboxes, scores, classIndices] = await yoloV3.detectFrame(
			imageObject
		);
		draw.renderOnImage(
			imageObject,
			selBboxes,
			scores,
			classIndices,
			classNames,
			imageObject.width,
			imageObject.height
		);
	});
	$('#runYoloSpinner').hide();
};

$(document).ready(function () {
	// disable button before any model loaded
	$('#runYolo').attr('disabled', true);
	// disable spinners
	$('#loadingModelSpinner').hide();
	$('#runYoloSpinner').hide();

	// init renderer:
	const font = configRender.font;
	const lineWidth = configRender.lineWidth;
	const lineColor = configRender.lineColor;
	const textColor = configRender.textColor;
	const textBackgoundColor = configRender.textBackgoundColor;

	const draw = new Render(
		canvas,
		lineWidth,
		lineColor,
		font,
		textColor,
		textBackgoundColor
	);
	// init model selection radio buttons:
	const models = Object.keys(modelsTable);
	models.map((option, index) => {
		$('#divRadioSelectModel')
			.append(
				$('<input>')
					.prop({ type: 'radio', id: option, name: 'model', value: option })
					.change((event) => onSelectModel(event))
			)
			.append(
				$('<label>')
					.prop({
						for: option,
					})
					.text(option)
			)
			.append($('<br>'));
	});
	// assign run button cbk
	$('#runYolo').click(() => onClickRunYolo(yoloV3, draw, imageUrl));
});
