import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"C:\Users\Mikail\.conda\envs\movies\data\movies_metadata.csv",low_memory=False)
df
df.info()

#henüz çıkmamış filmleri kaldır
df = df.query("status == 'Released'")

#gereksiz kolonları kaldır
df = df.drop(columns=['adult','homepage','imdb_id','poster_path','status','tagline','original_title','video'])
df.info()

#belongs_to_collection kolonunda nullları 0, dolu olanları 1 yapar
df['belongs_to_collection'] = df['belongs_to_collection'].notna().astype(int)

df.isnull().sum()

df = df.dropna(subset=['budget', 'revenue', 'release_date'])

df['budget'] = pd.to_numeric(df['budget'], errors='coerce') #budget kolonunu numeric tipe dönüştür
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
df.isnull().sum()

#date'i yalnızca yıl değerini alıp ayrı kolon olarak ekler
df.loc[:, 'release_year'] = df['release_date'].dt.year
df.info()


#filmlerin yayınlanma yıllarından aldığı verilerle hangi yıl kaç film çıkmış hesaplar
year_counts = df['release_year'].value_counts().sort_index()

plt.figure(figsize=(12,6))
plt.plot(year_counts.index, year_counts.values, marker='o')
plt.title("Yıllara Göre Çıkan Filmler", fontsize=16)
plt.xlabel("Yıl", fontsize=12)
plt.ylabel("Film Sayısı", fontsize=12)
plt.grid()
plt.show()


import ast
# 'genres' sütununu parse eder (json formatı halindeki veriden name değerini ayrıştırır)
def parse_genres(genres_str):
    try:
        genres = ast.literal_eval(genres_str)
        return [genre['name'] for genre in genres]
    except:
        return []

df['genres_parsed'] = df['genres'].apply(parse_genres)

#en çok hasılat kazanan 20 filmi listeler
top_revenue = df.nlargest(20, 'revenue')[['title', 'revenue']]

plt.figure(figsize=(10, 8))
plt.barh(top_revenue['title'], top_revenue['revenue'], color='skyblue')
plt.xlabel('Revenue (in billions)', fontsize=12)
plt.ylabel('Film Title', fontsize=12)
plt.title('Top 20 Movies by Revenue', fontsize=15)
plt.gca().invert_yaxis()  # Film isimlerini yukarıdan aşağı sırala
plt.tight_layout()
plt.show()


from collections import Counter

#tüm film tür değerlerini alır ve toplar
all_genres = [genre for genres in df['genres_parsed'] for genre in genres]
genre_counts = Counter(all_genres)

from matplotlib import cm

# Küçük dilimleri "Other" kategorisine eklemek için bir eşik belirleyelim
threshold = 0.03  # %5 altındaki türler "Other" kategorisine dahil edilecek
total_count = sum(genre_counts.values())

# Türleri ve frekanslarını "Other" kategorisine göre düzenler
filtered_genres = {k: v for k, v in genre_counts.items() if v / total_count >= threshold}
other_count = sum(v for k, v in genre_counts.items() if v / total_count < threshold)
if other_count > 0:
    filtered_genres["Other"] = other_count

# Verileri ayarla
labels = list(filtered_genres.keys())
sizes = list(filtered_genres.values())
colors = cm.tab20c(range(len(labels)))  # Renk paleti

# Veri
top_revenue = df.nlargest(20, 'revenue')[['genres_parsed', 'title', 'revenue']]
top_revenue['genres_parsed'] = top_revenue['genres_parsed'].apply(
    lambda x: ', '.join(x) if isinstance(x, list) else str(x)
)



# Çıkıntılı bir dilim eklemek için
explode = [0.1 if label == "Drama" else 0 for label in labels]

# Pasta grafiği güncelle
plt.pie(
    sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors,
    wedgeprops={'edgecolor': 'white'}, explode=explode, textprops={'fontsize': 10, 'color': 'white'}
)

# Başlık ekle
plt.title('Filmlerin Tür Dağılımları', fontsize=16)
plt.tight_layout()
plt.show()


# genres_parsed'deki aynı satırda bulunan birden fazla tür değerini işleyebilmek için ayırır
df_exploded = df.explode('genres_parsed')

#film türlerine göre kazanılan toplam geliri hesaplar
genre_revenue = df_exploded.groupby('genres_parsed')['revenue'].sum().sort_values(ascending=False)

genre_revenue.plot(kind='bar', figsize=(10, 6))
plt.title('Türlere Göre Toplam Gelir Grafiği')
plt.xlabel('Tür')
plt.ylabel('Toplam Gelir (yüz milyar)')
plt.grid(axis='y')
plt.show()

#yukarıdaki aynı işlemi toplam bütçe için tekrarlar
genre_budget = df_exploded.groupby('genres_parsed')['budget'].sum().sort_values(ascending=False)

genre_budget.plot(kind='bar', figsize=(10, 6))
plt.title('Türlere Göre Toplam Bütçe Grafiği')
plt.xlabel('Tür')
plt.ylabel('Toplam Bütçe (yüz milyar)')
plt.grid(axis='y')
plt.show()

# Gelir ve bütçe verileri
revenue = genre_revenue.values
budget = genre_budget.values

# Türler
genres = genre_revenue.index

# Bar grafiği için pozisyon ayarlama
x = np.arange(len(genres))  # Türlerin konumu
width = 0.35  # Çubuk genişliği

# Grafik oluşturma
plt.figure(figsize=(12, 6))
plt.bar(x - width/2, revenue, width, label='Gelir', color='skyblue')
plt.bar(x + width/2, budget, width, label='Bütçe', color='orange')

# Grafik detayları
plt.title('Türlere Göre Toplam Gelir ve Bütçe Karşılaştırması', fontsize=16)
plt.xlabel('Tür', fontsize=12)
plt.ylabel('Miktar (yüz milyar)', fontsize=12)
plt.xticks(x, genres, rotation=45, ha='right', fontsize=10)
plt.legend()
plt.grid(axis='y')
plt.tight_layout()
plt.show()


# Gelir/Bütçe oranı hesaplama
genre_ratio = genre_revenue / genre_budget

# Oran grafiği
genre_ratio.plot(kind='bar', figsize=(10, 6), color='green')
plt.title('Türlere Göre Gelir/Bütçe Oranı')
plt.xlabel('Tür')
plt.ylabel('Gelir/Bütçe Oranı')
plt.grid(axis='y')
plt.tight_layout()
plt.show()


# İlgili sütunları seçme
columns_of_interest = ['popularity', 'budget', 'revenue', 'vote_average', 'runtime','belongs_to_collection']
data_subset = df[columns_of_interest].copy()

# Sayısal sütunlara dönüştürme
for column in ['popularity', 'budget', 'revenue', 'vote_average', 'runtime','belongs_to_collection']:
    data_subset[column] = pd.to_numeric(data_subset[column], errors='coerce')

# Eksik değerleri doldurma (median veya başka bir yöntem kullanabilirsiniz)
data_subset = data_subset.dropna()

# Korelasyon matrisini hesaplama
correlation_matrix = data_subset.corr()
print(correlation_matrix)

# Isı haritası oluşturma
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Korelasyon Matrisi - Filmin Popülerliği ve Diğer Faktörler")
plt.show()



import plotly.express as px

def extract_countries(country_list):
    try:
        countries = ast.literal_eval(country_list)
        return ", ".join([country['name'] for country in countries])
    except:
        return None

# Ülkeleri çıkartma
df['countries'] = df['production_countries'].apply(extract_countries)

# Ülkelere göre toplam film sayısını hesaplama
country_counts = df['countries'].str.split(', ', expand=True).stack().value_counts().reset_index()
country_counts.columns = ['Country', 'Film Count']
# Log dönüşümü
country_counts['Log Film Count'] = np.log1p(country_counts['Film Count'])  # log(1 + x)



# Dünya haritası görselleştirme
fig = px.choropleth(
    country_counts,
    locations="Country",  # Ülke isimleri
    locationmode="country names",  # Harita modu: Ülke isimleri
    color="Log Film Count",  # Renk skalası: Film sayısı
    title="Ülkelere Göre Film Üretimi",
    color_continuous_scale="Blues",  # Renk skalası
)

# Ülke isimlerinin üzerine sayısal etiketler ekle
for i, row in country_counts.iterrows():
    fig.add_scattergeo(
        locations=[row['Country']],
        locationmode='country names',
        text=f"{row['Film Count']}",
        mode='text',
        textfont=dict(size=20, color='red'),
        showlegend=False
    )


fig.update_layout(
    geo=dict(showframe=False, showcoastlines=True, projection_type='equirectangular'),
)
fig.show()


#PREDICTION PART
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.preprocessing import MultiLabelBinarizer

# Eksik değerleri temizleme
data_cleaned = df[['budget', 'revenue', 'popularity', 'runtime', 'vote_average', 'release_date', 'genres']].dropna()

# NaN değerleri boş liste olarak doldurma
df['genres_parsed'] = df['genres_parsed'].apply(lambda x: eval(x) if isinstance(x, str) else [])

# Türleri binarize etme
mlb = MultiLabelBinarizer()
genres_encoded = mlb.fit_transform(df['genres_parsed'])

# Türleri genişletilmiş sütunlara dönüştürme
genres_df = pd.DataFrame(genres_encoded, columns=mlb.classes_)

# Orijinal veriyle birleştirme
df = pd.concat([df, genres_df], axis=1)


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Kullanılacak sütunları seçme
features = ['budget', 'popularity', 'runtime', 'vote_average'] + list(mlb.classes_)
target = 'revenue'

# Eksik verileri kaldırma ve bağımsız/bağımlı değişkenleri ayırma
data_cleaned = df[features + [target]].dropna()
X = data_cleaned[features]
y = data_cleaned[target]

# Veriyi eğitim ve test olarak ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model tanımlama ve eğitme
model = RandomForestRegressor(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

# Test verisinde tahmin yapma
y_pred = model.predict(X_test)

# Performans ölçütleri
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")


def predict_revenue(budget, popularity, runtime, vote_average, genres_list, model):
    """
    Belirtilen film özelliklerine göre tahmini gelir döndürür.
    """
    # Türleri binarize etme
    genres_vector = mlb.transform([genres_list]).flatten()

    # Özellikleri birleştirme
    features = [budget, popularity, runtime, vote_average] + list(genres_vector)

    # Tahmin yapma
    prediction = model.predict([features])
    return prediction[0]


# Örnek tahmin
sample_prediction = predict_revenue(
    budget=15000000,
    popularity=50,
    runtime=120,
    vote_average=7.5,
    genres_list=['Action', 'Adventure'],
    model=model
)
print(f"Tahmini Gelir: ${sample_prediction:.2f}")

# Örnek tahmin
sample_prediction = predict_revenue(
    budget=30000000,
    popularity=14.404764,
    runtime=89,
    vote_average=6.0,
    genres_list=['Fantasy', 'Drama','Comedy'],
    model=model
)
print(f"Tahmini Gelir: ${sample_prediction:.2f}")

# Örnek tahmin
sample_prediction = predict_revenue(
    budget=55000000,
    popularity=13.280069,
    runtime=81,
    vote_average=6.7,
    genres_list=['Adventure', 'Animation','Drama'],
    model=model
)
print(f"Tahmini Gelir: ${sample_prediction:.2f}")