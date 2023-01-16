import { YoloV3, createModel } from './YoloV3.js';
import Render from './Render.js';

import configModel from './configs/configModel.json' assert { type: 'json' };
import configNms from './configs/configNms.json' assert { type: 'json' };
import configRender from './configs/configRender.json' assert { type: 'json' };

$(document).ready(function () {
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
      'https://images.pexels.com/photos/13230453/pexels-photo-13230453.jpeg';
    // 'https://cdn.pixabay.com/photo/2017/09/04/19/02/scheepvaartmarkering-2715015_960_720.jpg';
    // 'https://images.pexels.com/photos/1181376/pexels-photo-1181376.jpeg';

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
  };

  demoYovoV3();
});
