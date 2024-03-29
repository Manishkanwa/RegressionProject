from flask import Flask, request, render_template, jsonify
from src.pipline.prediction_pipeline import custom_data, PredictPipeline

application = Flask(__name__)
app = application

@app.route("/")
def homepage():
    return render_template("index.html")

@app.route("/predict", methods= ["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template('form.html')
    else :
        data = custom_data(
            carat = float(request.form.get( 'carat' ) ) ,
            depth = float(request.form.get( 'depth' ) ) ,
            table = float(request.form.get( 'table' ) ) ,
            x = float(request.form.get( 'x')),
            y = float(request.form.get('y')),
            z = float(request.form.get('z')),
            cut = request.form.get( 'cut' ),
            color = request.form.get( 'color' ) ,
            clarity= request.form.get('clarity')
        )
        final_df = data.get_data_as_dataframe()
        predict_pipline = PredictPipeline()
        pred = predict_pipline.predictpipeline(final_df)
        results = round(pred[0], 2)
        return render_template("form.html", final_result=results)


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug= True)