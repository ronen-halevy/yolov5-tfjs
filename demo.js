import { YoloV3, createModel } from './YoloV3.js';
import Render from './Render.js';

// {
//   "title": "walking-in-university",
//   "url": "https://cdn.pixabay.com/photo/2017/09/04/19/02/scheepvaartmarkering-2715015_960_720.jpg"
// },
// {
//   "title": "cyclists",
//   "url": "https://images.pexels.com/photos/13230453/pexels-photo-13230453.jpeg"
// },
// {
//   "title": "walking-in-field",
//   "url": "https://images.pexels.com/photos/1181376/pexels-photo-1181376.jpeg"
// },
// {
//   "title": "dog-sitting",
//   "url": "https://cdn.pixabay.com/photo/2023/01/01/16/35/street-7690347_960_720.jpg"
// },
// {

import configModel from './configs/configModel.json' assert { type: 'json' };
import configNms from './configs/configNms.json' assert { type: 'json' };
import configRender from './configs/configRender.json' assert { type: 'json' };

import cocoExamples from './examples/cocoExamples.json' assert { type: 'json' };

$(document).ready(function () {
  var model = '';
  var anchors = '';
  var classNames = '';
  console.log(classNames);

  const modelsTable1 = configModel.models;
  const models = Object.keys(modelsTable1);
  var selectedModel = Object.keys(modelsTable1)[0];

  var selectedWeights = Object.keys(modelsTable1[selectedModel])[0];

  const onLoadModel = async () => {
    $('#waitLoadingModel').show();

    console.log('gggggggggggggg', selectedModel, selectedWeights);
    console.log('gggggggggggggg', modelsTable1);

    const { modelUrl, anchorsUrl, classNamesUrl } =
      modelsTable1[selectedModel][selectedWeights];

    [model, anchors, classNames] = await createModel(
      modelUrl,
      anchorsUrl,
      classNamesUrl
    );
    console.log(anchors);
    classNames = classNames.split(/\r?\n/);

    $('#waitLoadingModel').hide();

    console.log(model, anchors, classNames);
  };
  $('#waitLoadingModel').hide();

  // listExamples.map((option, index) => (
  // )
  // $('#submit').click(function () {
  $('#loadModel').html('Load Model');
  $('#loadModel').click(onLoadModel);

  var scaleFactor = 1;
  $('#scale').html('x' + scaleFactor);
  $('#scale').click(() => {
    scaleFactor = scaleFactor * 2 >= 1 ? 0.25 : scaleFactor * 2;
    $('#scale').html('x' + scaleFactor);
    console.log('hhhhh');
  });

  const onChangeWןwights = (event) => {
    selectedWeights = event.target.value;
  };
  const onChangeRadio = (event) => {
    selectedModel = event.target.value;

    dislayWeightsButtons(selectedModel);
  };
  const dislayWeightsButtons = (selectedModel) => {
    selectedWeights = Object.keys(modelsTable1[selectedModel]);
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
          .change(onChangeRadio)
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

  // var r = $('<input type="button" value="select model"/>');
  // $('#selectExample').append(r);

  console.log(cocoExamples.cocoImages);
  const cocoImages = cocoExamples.cocoImages;
  var selectedExample = cocoImages[0];
  var exampleUrl = selectedExample.url;
  $('#selectedExampleTitle').html(selectedExample.title);

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

  // .change((event) => {
  //   console.log(event);
  // });

  // $('#selectModel').on('change', function () {
  //   alert(this.value);
  //   alert(this.value);
  // });

  // models.map((option, index) => {
  //   $('#selectModel').append(new Option(option, index));
  // });

  // $('#selectModel').on('change', function () {
  //   alert(this.value);
  //   alert(this.value);
  // });
  $('#runYolo').html('Run Yolo');
  $('#runYolo').click(async () => {
    console.log(model);
    console.log(anchors);
    console.log(classNames);

    const modelsTable = configModel.models;
    // take first in list as a default:
    // const selectedModel = Object.keys(modelsTable)[0];
    // const selectedWeights = Object.keys(modelsTable[selectedModel])[0];
    // const modelParams = configModel['models'][selectedModel][selectedWeights];

    // const { modelUrl, anchorsUrl, classNamesUrl } = modelParams;
    // // const
    // let [model, anchors, classNames] = await createModel(
    //   modelUrl,
    //   anchorsUrl,
    //   classNamesUrl
    // );
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
