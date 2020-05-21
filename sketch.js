let r,g,b;
let data;
let xdata = [];
let ydata = [];
let x;
let y;
let changecolbtn;
let predictcolbtn;
trainingtxt = "training...";
let nn;
let predictiontxt = "";

function preload() {
  data = loadJSON("colourdata.json");

}
function setup() {
  createCanvas(600,600);

  r = floor(random(255));
  g = floor(random(255));
  b = floor(random(255));

let changecolbtn = createButton("Change colour");
changecolbtn.position(200, 400);
let predictcolbtn = createButton("Predict colour");
predictcolbtn.position(changecolbtn.x+100,400)



  for (let i = 0; i < data.x.length; i++) {
    xdata[i] = data.x[i];

  }
  for (let i = 0; i < data.y.length; i++) {
    ydata[i] = data.y[i];

  }

  let x = tf.tensor(xdata);
  let y = tf.tensor(ydata);


  changecolbtn.mousePressed(pickColor);

  nn = tf.sequential();

  let hidden = tf.layers.dense({
    units: 10,
    activation: "sigmoid",
    inputShape: [3]
  });
  let output = tf.layers.dense({
    units: 11,
    activation: "softmax"

  });
  nn.add(hidden);
  nn.add(output);

  nn.compile({
    optimizer: tf.train.sgd(0.2),
    loss: "meanSquaredError"



  });


  async function train() {
    for (let i = 0; i < 100; i++) {
      const response = await nn.fit(x,y,{shuffle:true, epochs:10});
      console.log("epoch: " + (i+1) + " / 100" + "\n loss: " + response.history.loss[0]);



    }


  }
  console.log("training...");

  train().then(() => {
    console.log("training complete");
    trainingtxt = "training complete";



  });
  predictcolbtn.mousePressed(predictColor);





}
function draw() {
  background(r,g,b);
  fill(255);
  noStroke();
  textAlign(CENTER, CENTER);

  text(trainingtxt, 300,300);

  text(predictiontxt, 300, 500);
  

}
function pickColor() {
  if (trainingtxt != "training...") {
    r = floor(random(255));
    g = floor(random(255));
    b = floor(random(255));
    trainingtxt = "";

  }



}
function predictColor() {
  if (trainingtxt != "training...") {
    let prediction = nn.predict(tf.tensor([[r,g,b]]));
    prediction = prediction.arraySync();
    let maxp = indexOfMax(prediction[0]);


    trainingtxt = "";
    textSize(40);
    if (maxp == 0) {
      predictiontxt = "Red";

    }
    if (maxp == 1) {
      predictiontxt = "Green";

    }
    if (maxp == 2) {
      predictiontxt = "Blue";

    }
    if (maxp == 3) {
      predictiontxt = "Yellow";

    }
    if (maxp == 4) {
      predictiontxt = "Orange";

    }
    if (maxp == 5) {
      predictiontxt = "Pink";

    }
    if (maxp == 6) {
      predictiontxt = "Purple";

    }
    if (maxp == 7) {
      predictiontxt = "Brown";

    }
    if (maxp == 8) {
      predictiontxt = "Grey";

    }
    if (maxp == 9) {
      predictiontxt = "White";

    }
    if (maxp == 10) {
      predictiontxt = "Black";

    }


  }

}
function indexOfMax(arr) {
    if (arr.length === 0) {
        return -1;
    }

    var max = arr[0];
    var maxIndex = 0;

    for (var i = 1; i < arr.length; i++) {
        if (arr[i] > max) {
            maxIndex = i;
            max = arr[i];
        }
    }

    return maxIndex;
}
