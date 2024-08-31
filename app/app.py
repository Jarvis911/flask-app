from flask import Flask, request, render_template, make_response
from model import load_and_preprocess_data, recommend_items_for_products

app = Flask(__name__)

# Khởi tạo model
# svd_model, product_list, data, df, scaler, scaler2 = load_and_preprocess_data("D:\\WorkSpaceVS\\AppTest\\data\\Master.csv")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Sửa filepath khi chạy chương trình
    svd_model, product_list, data, df, scaler, scaler2 = load_and_preprocess_data("D:\\WorkSpaceVS\\AppTest\\data\\Master.csv")
    product_names = [request.form.get('product1'), request.form.get('product2'), request.form.get('product3')]
    
    recommendations = recommend_items_for_products(product_names, svd_model, product_list, data, df, scaler, scaler2)
    recommendations = recommendations.drop_duplicates(subset=['product_name'])
    
    response = make_response(render_template('index.html', recommendations=recommendations.to_dict('records')))
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    
    return response

if __name__ == '__main__':
    app.run(debug=True)