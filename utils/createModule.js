/**
* @file Brief description of the file here

Prepares a duplicate of the input file, remove export line and appends modules exports
The created file can be used as a cdn module. Note: Otherwise, gh-pages blocks cdn dynamic import 
# Input and output file names are ghardcoded - tbd.

# The target line to remove is stored in a var named removeLine 
# The module.exports line to be appended is stored in a var named newLine

# Source file is stored in a variable named src
# Dest file is stored in a variable named dst

# Comand line:
run:
node createModule.js



* @author FirstName LastName <optionalEmail@example.com>
* @copyright FirstName LastName Year
* @license LicenseHereIfApplicable

*/

const createModuleFile = (src, dst, removeLine, newLine, fs) => {
	// 1. Remove line
	fs.readFile(src, { encoding: 'utf-8' }, function (err, data) {
		if (err) throw error;

		let dataArray = data.split('\n'); // convert file data in an array
		let lastIndex = -1;

		for (let index = 0; index < dataArray.length; index++) {
			if (dataArray[index].includes(removeLine)) {
				// check if a line contains the 'user1' keyword
				lastIndex = index; // found a line includes a 'user1' keyword
				break;
			}
		}

		// write to new file
		dataArray.splice(lastIndex, 1);

		const updatedData = dataArray.join('\n');
		console.log('write!!!!');
		fs.writeFile(dst, updatedData, (err) => {
			if (err) throw err;
			console.log('Successfully updated the file data');
			// append new line:
			fs.open(dst, 'a', 666, function (e, id) {
				fs.write(id, newLine + '\r\n', null, 'utf8', function () {
					fs.close(id, function () {
						console.log('file is updated');
					});
				});
			});
		});
	});
};

var src = 'YoloV5.js';
var dst = 'YoloV5.module.js';
var fs = require('fs');

// line to remove:
var removeLine = 'export { YoloV5, createModel };';
var newLine =
	'const yolov5 = {YoloV5: YoloV5, createModel: createModel} \nmodule.exports = yolov5';
createModuleFile(src, dst, removeLine, newLine, fs);

var src = 'Render.js';
var dst = 'Render.module.js';

// line to remove:
var removeLine = 'export { Render };';
var newLine = 'const render = {Render: Render} \nmodule.exports = render';
createModuleFile(src, dst, removeLine, newLine, fs);
