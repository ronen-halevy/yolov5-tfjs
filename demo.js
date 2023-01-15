import { YoloV3, createModel } from './YoloV3.js';
import Render from './Render.js';

import configModel from './configs/configModel.json' assert { type: 'json' };
import configNms from './configs/configNms.json' assert { type: 'json' };
import configRender from './configs/configRender.json' assert { type: 'json' };

const demoYovoV3 = async () => {
  const modelsTable = configModel.models;
  // take first in list as a default:
  const selectedModel = Object.keys(modelsTable)[0];

  const selectedWeights = Object.keys(modelsTable[selectedModel])[0];

  const modelParams = configModel['models'][selectedModel][selectedWeights];

  const { modelUrl, anchorsUrl, classNamesUrl } = modelParams;
  // const
  let [model, anchors, classNames] = await createModel(
    modelUrl,
    anchorsUrl,
    classNamesUrl
  );

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

  const canvas = document.getElementById('canvas');

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

  const imgUrl =
    'https://www.shutterstock.com/shutterstock/photos/290439/display_1500/stock-photo-two-labrador-retriever-puppies-following-their-mother-down-a-country-road-290439.jpg';

  var imageObject = new window.Image();

  const res = await fetch(imgUrl);
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
};

demoYovoV3();
