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

$(document).ready(function () {
  var model = '';
  var anchors = '';
  var classNames = '';
  // var r = $('<input type="button" value="select model"/>');
  // $('#buttons').append(r);
  const modelsTable1 = configModel.models;
  const models = Object.keys(modelsTable1);
  var selectedModel = Object.keys(modelsTable1)[0];

  var selectedWeights = Object.keys(modelsTable1[selectedModel])[0];

  const onLoadModel = async () => {
    console.log('gggggggggggggg', selectedModel, selectedWeights);
    const { modelUrl, anchorsUrl, classNamesUrl } =
      modelsTable1[selectedModel][selectedWeights];

    [model, anchors, classNames] = await createModel(
      modelUrl,
      anchorsUrl,
      classNamesUrl
    );
    console.log(model, anchors, classNames);
  };
  // listExamples.map((option, index) => (
  // )
  // $('#submit').click(function () {
  $('#loadModel').html('Load Model');
  $('#loadModel').click(onLoadModel);

  const onChangeWןwights = (event) => {
    selectedWeights = event.target.value;
  };
  const onChangeRadio = (event) => {
    console.log('ddddd', event.target.value, models, models[selectedModel]);
    selectedModel = event.target.value;

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

  // models.map((option, index) => {
  //   $('#selectModel').append(new Option(option, index));
  // });

  // $('#selectModel').on('change', function () {
  //   alert(this.value);
  //   alert(this.value);
  // });
  $('#runYolo').html('Run Yolo');
  $('#runYolo').click(async () => {
    console.log(model, anchors, classNames);
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
    classNames = classNames.split(/\r?\n/);
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

    const imgUrl1 =
      'https://images.pexels.com/photos/13230453/pexels-photo-13230453.jpeg';

    const imgUrl =
      // 'https://images.pexels.com/photos/13230453/pexels-photo-13230453.jpeg';
      // 'https://cdn.pixabay.com/photo/2017/09/04/19/02/scheepvaartmarkering-2715015_960_720.jpg';
      // 'https://images.pexels.com/photos/1181376/pexels-photo-1181376.jpeg';
      'https://images.pexels.com/photos/1181376/pexels-photo-1181376.jpeg';

    var imageObject = new window.Image();

    const res = await fetch(imgUrl);
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
        imageObject.width,
        imageObject.height
      );
    });
  });

  // demoYovoV3();
});
