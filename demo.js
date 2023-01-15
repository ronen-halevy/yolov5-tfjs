import { YoloV3, createModel } from './YoloV3.js';
import Render from './Render.js';

import configModel from './configs/configModel.json' assert { type: 'json' };
import configNms from './configs/configNms.json' assert { type: 'json' };

const modelsTable = configModel.models;
// take first in list as a default:
const selectedModel = Object.keys(modelsTable)[0];
console.log(selectedModel);

const selectedWeights = Object.keys(modelsTable[selectedModel])[0];
console.log(selectedWeights);
console.log(configModel);

const modelParams = configModel['models'][selectedModel][selectedWeights];

const { modelUrl, anchorsUrl, classNamesUrl } = modelParams;
console.log(modelUrl, anchorsUrl, classNamesUrl);
// const
let [model, anchors, classNames] = await createModel(
  modelUrl,
  anchorsUrl,
  classNamesUrl
);

classNames = classNames.split(/\r?\n/);
const nClasses = classNames.length;
console.log([classNames]);
console.log(nClasses);

const { scoreTHR, iouTHR, maxBoxes } = configNms;

const yoloV31 = new YoloV3(
  model,
  anchors.anchor,
  nClasses,
  scoreTHR,
  iouTHR,
  maxBoxes
);

const imgUrl1 =
  'https://www.shutterstock.com/shutterstock/photos/1262270857/display_1500/stock-photo-curvy-windy-road-in-snow-covered-forest-top-down-aerial-view-1262270857.jpg';

var imageObject = new window.Image();

const res = await fetch(imgUrl1);
const imageBlob = await res.blob();
const imageObjectURL = URL.createObjectURL(imageBlob);
imageObject.src = imageObjectURL;
imageObject.addEventListener('load', async () => {
  const f = yoloV31.detectFrame(imageObject).then((res) => {
    console.log(res);
  });
});

console.log(selectedWeights);
