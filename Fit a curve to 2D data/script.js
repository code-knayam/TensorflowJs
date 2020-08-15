function testModel(model, inputData, normalizationData) {
	const {inputMax, inputMin, labelMax, labelMin} = normalizationData;

	// Generate predictions for a uniform range of numbers between 0 and 1;
  // We un-normalize the data by doing the inverse of the min-max scaling 
  // that we did earlier.
	const [xs, preds] = tf.tidy(() => {

		const xs = tf.linspace(0, 1, 100);
		const preds = model.predict(xs.reshape([100, 1]));

		const unNormXs = xs.mul(inputMax.sub(inputMin)).add(inputMin);

		const unNormPreds = preds.mul(labelMax.sub(labelMin)).add(labelMin);

		return [unNormXs.dataSync(), unNormPreds.dataSync()]
	});

	const predictedPoints = Array.from(xs).map((val, i) => {
		return {x: val, y: preds[i]};
	});

	const originalPoints = inputData.map(d => ({
		x: d.horsepower, y: d.mpg
	}))

	tfvis.render.scatterplot(
	{name: 'Model predictions vs original data'},
	{values: [originalPoints, predictedPoints], series: ['original', 'predicted' ]},
	{
		xLabel: 'Horsepower',
		yLabel: 'MPG',
		height: 300
	}
	);
}



async function trainModel(model, inputs, labels) {
	// prepare the model for training
	model.compile({
		optimizer: tf.train.adam(),
		loss: tf.losses.meanSquaredError,
		metrics: ['mse']
	});

	const batchSize = 32;
	const epochs = 50;

	return await model.fit(inputs, labels, {
		batchSize, 
		epochs,
		shuffle: true,
		callbacks: tfvis.show.fitCallbacks(
			{name: 'Training Performance'},
			['loss', 'mse'],
			{height: 200, callbacks: ['onEpochEnd']}
		)
	})
}

/**
 * Convert the input data to tensors that we can use for machine 
 * learning. We will also do the important best practices of _shuffling_
 * the data and _normalizing_ the data
 * MPG on the y-axis.
 */
function converToTensor(data) {
	// wrappng these calculations in a tidy will dispose any intermediate tensors

	return tf.tidy(() => {
		// step 1 shuffle the data
		tf.util.shuffle(data);

		// step 2 conver data to tensor
		const inputs = data.map(d => d.horsepower);
		const labels = data.map(d => d.mpg);

		const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
		const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

		// step 3 normalize the data to the range 0-1 using min-max scaling
		const inputMax = inputTensor.max();
		const inputMin = inputTensor.min();
		const labelMax = labelTensor.max();
		const labelMin = labelTensor.min();

		const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
		const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));
		

		return {
			inputs: normalizedInputs, 
			labels: normalizedLabels,
			// return min/max bounds
			inputMax,
			inputMin, 
			labelMax,
			labelMin
		}
	});
}


function createModel() {
	const model = tf.sequential();

	model.add(tf.layers.dense({inputShape: [1], units: 1, useBias: true}));
	// model.add(tf.layers.dense({units: 50, activation: 'sigmoid'}));

	model.add(tf.layers.dense({units: 1, useBias: true}));

	return model;
}

async function getData() {
	const carsDataReq = await fetch('https://storage.googleapis.com/tfjs-tutorials/carsData.json');
	const carsData = await carsDataReq.json();
	const cleaned = carsData.map(car => ({
		mpg: car.Miles_per_Gallon,
		horsepower : car.Horsepower
	}))
	.filter(car => (car.mpg != null && car.horsepower != null));

	return cleaned;
}

async function run() {
	const data = await getData();
	const values = data.map(d => ({
		x: d.horsepower,
		y: d.mpg
	}));

	tfvis.render.scatterplot(
		{name: 'Hotsepower v MPG'},
		{values},
		{
			xLabel: 'Horsepower',
			yLabel: 'MPG',
			height: 300
		}
	);	

	const model = createModel();
	tfvis.show.modelSummary({name: 'Model Summary'}, model);

	const tensorData = converToTensor(data);
	const {inputs, labels} = tensorData;

	await trainModel(model, inputs, labels);
	console.log('training complete');

	testModel(model, data, tensorData);
}

document.addEventListener('DOMContentLoaded', run);