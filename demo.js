import { YoloV3, createModel } from './YoloV3.js';
import Render from './Render.js';

import configModel from './configs/configModel.json' assert { type: 'json' };
import configNms from './configs/configNms.json' assert { type: 'json' };
import configRender from './configs/configRender.json' assert { type: 'json' };

import cocoExamples from './examples/cocoExamples.json' assert { type: 'json' };

$(document).ready(function () {
	// Model Load Functions
	const loadModel = async (modelsTable, selectedModel, selectedWeights) => {
		$('#waitLoadingModel').show();

		const { modelUrl, anchorsUrl, classNamesUrl } =
			modelsTable[selectedModel][selectedWeights];

		[model, anchors, classNames] = await createModel(
			modelUrl,
			anchorsUrl,
			classNamesUrl
		);
		console.log(anchors);
		classNames = classNames.split(/\r?\n/);

		$('#waitLoadingModel').hide();

		$('#loadedModelTilte').text(
			'Loaded: ' + selectedModel + '+' + selectedWeights
		);
	};

	var model = '';
	var anchors = '';
	var classNames = '';
	console.log(classNames);

	const modelsTable = configModel.models;
	const models = Object.keys(modelsTable);
	var selectedModel = Object.keys(modelsTable)[0];
	var selectedWeights = Object.keys(modelsTable[selectedModel])[0];

	const onLoadModel = async () => {
		await loadModel(modelsTable, selectedModel, selectedWeights);
	};
	$('#waitLoadingModel').hide();
	onLoadModel();

	$('#loadModel').text('Load Model');
	$('#loadModel').click(onLoadModel);

	var scaleFactor = 0.125;
	$('#scale').text('x' + scaleFactor);
	$('#scale').click(() => {
		scaleFactor = scaleFactor * 2 > 1 ? 0.125 : scaleFactor * 2;
		$('#scale').text('x' + scaleFactor);
	});

	const onChangeWןwights = (event) => {
		selectedWeights = event.target.value;
	};
	const onChangeModelSelect = (event) => {
		selectedModel = event.target.value;

		dislayWeightsButtons(selectedModel);
	};
	const dislayWeightsButtons = (selectedModel) => {
		selectedWeights = Object.keys(modelsTable[selectedModel]);
		$('#divRadioSelectWeights').empty();
		selectedWeights.map((option, index) => {
			console.log(option, index);
			$('#divRadioSelectWeights')
				.append(
					$('<input>')
						.prop({
							type: 'radio',
							id: option,
							name: 'weights',
							value: option,
						})
						.change(onChangeWןwights)
				)
				.append(
					$('<label>')
						.prop({
							for: option,
						})
						.html(option)
				)
				.append($('<br>'));
		});
		console.log('ddddd', selectedWeights);
	};

	models.map((option, index) => {
		console.log(option, index);
		$('#divRadioSelectModel')
			.append(
				$('<input>')
					.prop({
						type: 'radio',
						id: option,
						name: 'model',
						value: option,
					})
					.change(onChangeModelSelect)
			)
			.append(
				$('<label>')
					.prop({
						for: option,
					})
					.html(option)
			)
			.append($('<br>'));
	});

	$("input[id|='YoloV3Tiny']").attr('checked', true);

	// $('#YoloV3').attr('checked', true);
	dislayWeightsButtons('YoloV3Tiny');
	$("input[id|='coco']").attr('checked', true);

	console.log(cocoExamples.cocoImages);
	const cocoImages = cocoExamples.cocoImages;
	var selectedExample = cocoImages[0];
	var exampleUrl = selectedExample.url;
	$('#selectedExampleTitle').html('Title: ' + selectedExample.title);

	cocoImages.map((option, index) => {
		console.log(option);
		$('#selectExample').append(new Option(option.url, index));
	});
	$('#selectExample').change((event) => {
		console.log(event.target.value);
		selectedExample = cocoImages[event.target.value];
		exampleUrl = selectedExample.url;

		console.log(selectedExample);
		$('#selectedExampleTitle').html(selectedExample.title);
		// selecteTitle = cocoImages[event.target.value].title;
	});

	$('#runYolo').html('Run Yolo');
	$('#runYolo').click(async () => {
		console.log(model);
		console.log(anchors);
		console.log(classNames);

		const nClasses = classNames.length;

		const { scoreTHR, iouTHR, maxBoxes } = configNms;

		const yoloV3 = new YoloV3(
			model,
			anchors.anchor,
			nClasses,
			scoreTHR,
			iouTHR,
			maxBoxes
		);
		console.log(model);

		// const canvas = document.getElementById('canvas');
		const canvas = $('#canvas')[0];

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

		var imageObject = new window.Image();
		console.log(exampleUrl);
		const res = await fetch(exampleUrl);
		const imageBlob = await res.blob();
		const imageObjectURL = URL.createObjectURL(imageBlob);
		imageObject.src = imageObjectURL;
		imageObject.addEventListener('load', async () => {
			const [selBboxes, scores, classIndices] = await yoloV3.detectFrame(
				imageObject
			);
			console.log(selBboxes);

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
	});

	// demoYovoV3();
});
