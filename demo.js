import { YoloV3, createModel } from './YoloV3.js';
import Render from './Render.js';

import configModel from './configs/configModel.json' assert { type: 'json' };
import configNms from './configs/configNms.json' assert { type: 'json' };
import configRender from './configs/configRender.json' assert { type: 'json' };

import cocoExamples from './examples/cocoExamples.json' assert { type: 'json' };

const font = configRender.font;
const lineWidth = configRender.lineWidth;
const lineColor = configRender.lineColor;
const textColor = configRender.textColor;
const textBackgoundColor = configRender.textBackgoundColor;

// var model = '';
// var anchors = '';
var yoloV3 = '';
var classNames = '';

const modelsTable = configModel.models;
const models = Object.keys(modelsTable);
var selectedModel = Object.keys(modelsTable)[1];
var selectedWeights = Object.keys(modelsTable[selectedModel])[0];

// const canvas = $('#canvas')[0];
const draw = new Render(
	canvas,
	lineWidth,
	lineColor,
	font,
	textColor,
	textBackgoundColor
);
const loadModel = async (selModel, selWeights) => {
	$('#waitLoadingModel').show();

	const { modelUrl, anchorsUrl, classNamesUrl } =
		modelsTable[selModel][selWeights];

	const [model, anchors, classNamesString] = await createModel(
		modelUrl,
		anchorsUrl,
		classNamesUrl
	);
	classNames = classNamesString.split(/\r?\n/);

	$('#waitLoadingModel').hide();

	$('#loadedModelTilte').text('Loaded: ' + selModel + '+' + selWeights);
	return [model, anchors, classNames];
};
const createDetector = async (selModel, selWeights) => {
	const [model, anchors, classNames] = await loadModel(selModel, selWeights);
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
};

const createModelSelectElements = (createWeightsElementsCallback) => {
	// model select butttons
	models.map((option, index) => {
		$('#divRadioSelectModel')
			.append(
				$('<input>')
					.prop({
						type: 'radio',
						id: option,
						name: 'model',
						value: option,
					})
					.change((event) => {
						selectedModel = event.target.value;
						createWeightsElementsCallback(event.target.value);
					})
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
};

const createWeightsElements = (selectedModel) => {
	const weights = Object.keys(modelsTable[selectedModel]);
	$('#divRadioSelectWeights').empty();
	weights.map((option, index) => {
		$('#divRadioSelectWeights')
			.append(
				$('<input>')
					.prop({
						type: 'radio',
						id: option,
						name: 'weights',
						value: option,
					})
					.change((event) => {
						selectedWeights = event.target.value;
					})
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
};
const runDetector = async (imageUrl, scaleFactor) => {
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
			imageObject.width * scaleFactor,
			imageObject.height * scaleFactor
		);
	});
};

$(document).ready(function () {
	// disable spinners:
	$('#waitLoadingModel').hide();
	$('#waitYolo').hide();

	// arrange model. Note set  createWeightsElements callback:
	createModelSelectElements(createWeightsElements);

	// todo - create detector with parameter so init is mode clear???
	// arrange modell load elements:
	$('#loadModel').text('Load Model');
	$('#loadModel').click((event) =>
		createDetector(selectedModel, selectedWeights)
	);

	// Load model and detector with default selections on init:
	createWeightsElements(selectedModel);
	createDetector(selectedModel, selectedWeights);
	// mark selected buttons:
	$('#' + selectedModel).attr('checked', true);
	$('#' + selectedWeights).attr('checked', true);

	// Select InputExample elements
	const cocoImages = cocoExamples.cocoImages;
	var selectedExample = cocoImages[0];
	var exampleUrl = selectedExample.url;
	$('#selectedExampleTitle').text('Title: ' + selectedExample.title);
	cocoImages.map((option, index) => {
		$('#selectExample').append(new Option(option.url, index));
	});
	$('#selectExample').change((event) => {
		selectedExample = cocoImages[event.target.value];
		exampleUrl = selectedExample.url;

		$('#selectedExampleTitle').text(selectedExample.title);
		// selecteTitle = cocoImages[event.target.value].title;
	});

	// arrange scale factor elements:
	var scaleFactor = 0.125;
	$('#scale').text('x' + scaleFactor);
	$('#scale').click(() => {
		scaleFactor = scaleFactor * 2 > 1 ? 0.125 : scaleFactor * 2;
		$('#scale').text('x' + scaleFactor);
	});

	$('#runYolo').text('Run Yolo');
	$('#runYolo').click(async () => {
		$('#waitYolo').show();

		await runDetector(exampleUrl, scaleFactor);

		$('#waitYolo').hide();
	});
});
