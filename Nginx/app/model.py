# model.py
import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler

def load_and_preprocess_data(file_path='Master.csv'):
    
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path, low_memory=False)
    df = df[df['status'] != 'Cancelled']

    product_list = df[['product_id', 'product_name', 'brand', 'category', 'sale_price']].drop_duplicates()

    data = df[['user_id', 'product_id', 'sale_price', 'num_of_item', 'category', 'brand']]
    data = data.reset_index(drop=True)
    data['sale_price'] = data['sale_price'].astype('float32')
    data = data.drop_duplicates()
    data['total_spend'] = data['sale_price'] * data['num_of_item']

    label_encoder_category = LabelEncoder()
    label_encoder_brand = LabelEncoder()
    data['category'] = label_encoder_category.fit_transform(data['category']).astype('int32')
    data['brand'] = label_encoder_brand.fit_transform(data['brand']).astype('int32')
    data['num_of_item'] = data['num_of_item'].astype('int32')

    scaler = StandardScaler()
    scaler2 = StandardScaler()
    data[['user_id', 'product_id', 'sale_price', 'num_of_item']] = scaler.fit_transform(data[['user_id', 'product_id', 'sale_price', 'num_of_item']])
    data[['total_spend']] = scaler2.fit_transform(data[['total_spend']])

    data = data.sample(80000)
    reader = Reader(rating_scale=(0, 1))
    dataset = Dataset.load_from_df(data[['user_id', 'product_id', 'num_of_item']], reader)
    trainset, testset = train_test_split(dataset, test_size=0.2, random_state=42)

    svd_model = SVD(n_factors=100, n_epochs=100, lr_all=0.005, reg_all=0.02)
    svd_model.fit(trainset)

    # Trả về scaler và data
    return svd_model, product_list, data, df, scaler, scaler2

def recommend_items_for_products(product_names, model, product_list, data, df, scaler, scaler2, n=5):
    user_id = -1  # Sử dụng ID người dùng giả định để tạo dự đoán
    user_items = []

    # Lọc lấy product_id tương ứng với product_name
    for name in product_names:
        product_id = product_list[product_list['product_name'] == name]['product_id'].values
        if len(product_id) > 0:
            user_items.append(product_id[0])

    if not user_items:
        # Trường hợp không có sản phẩm nào hợp lệ, trả về DataFrame trống
        return pd.DataFrame()

    # Lấy danh sách tất cả các sản phẩm không thuộc user_items
    all_items = data['product_id'].unique()
    items_to_predict = np.setdiff1d(all_items, user_items)

    # Tạo dự đoán cho các sản phẩm này
    predictions = [model.predict(user_id, item) for item in items_to_predict]
    top_n = sorted(predictions, key=lambda x: x.est, reverse=True)[:n]

    # Tạo DataFrame kết quả dự đoán
    rec_df = pd.DataFrame([(rec.iid, round(rec.est, 4)) for rec in top_n], columns=['product_id', 'estimated_rating'])
    rec_df = rec_df.drop_duplicates(subset=['product_id'])

    # Khôi phục lại product_id gốc
    original_product_ids = scaler.inverse_transform(data[['user_id', 'product_id', 'sale_price', 'num_of_item']])[:, 1]
    product_id_map = dict(zip(data['product_id'], original_product_ids))
    rec_df['original_product_id'] = rec_df['product_id'].map(product_id_map)

    # Kết hợp với dữ liệu gốc để lấy thông tin sản phẩm
    original_data = df[['product_id', 'product_name', 'sale_price']]
    rec_df = rec_df.merge(original_data, left_on='original_product_id', right_on='product_id', how='left')

    return rec_df[['product_name', 'sale_price', 'estimated_rating']]

