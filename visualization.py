import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt

matplotlib.use('Qt5Agg', force=True)

# 데이터 로드
df = sns.load_dataset('diamonds')
df.info()
df.describe()
df.isna().sum()

# Scatter Plot (산점도)
sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 10))
color_palette = {
    'Fair': 'black',
    'Good': 'gold',
    'Very Good': 'mediumseagreen',
    'Premium': 'crimson',
    'Ideal': 'royalblue'
}
sns.scatterplot(data=df, x='carat', y='price', hue='cut', palette=color_palette, alpha=0.3, s=30)
plt.title("Carat vs. Price", fontsize = 20, fontweight = 'bold', pad=15)
plt.xlabel("Carat", fontsize = 14, color = 'dimgray', labelpad=15)
plt.ylabel("Price", fontsize = 14, color = 'dimgray', labelpad=15)
plt.tick_params(axis='both', which='major', labelsize=12, colors='gray')
plt.grid(True, linestyle='--', linewidth=1, color='gray', alpha=0.5)
sns.despine(trim=True, offset=5)
plt.legend(title='Cut', title_fontsize=10, fontsize=9, loc='upper left', frameon=False)
plt.tight_layout()
plt.show()

# Bar Plot (막대 그래프)
cfm = df[df['cut'] == 'Fair']['price'].mean()
sns.set_theme(style='whitegrid')
plt.figure(figsize=(10,8))
sns.barplot(x='cut', y='price', data=df, hue='cut', palette='plasma', dodge=False)
plt.axhline(y=cfm, color='red', linestyle='--', linewidth=1.2, alpha=0.7, label = 'Fair Cut Mean Price')
plt.legend(fontsize=12, loc='upper right', frameon=False)
plt.title("Mean Price by Cut", fontsize=18, fontweight='bold', pad=20)
plt.xlabel("Cut", fontsize=14, color='dimgray', labelpad=15)
plt.ylabel("Mean Price", fontsize=14, color='dimgray', labelpad=15)
plt.tick_params(axis='both', labelsize=12, colors='gray')
sns.despine()
plt.tight_layout()
plt.show()

# Horizontal Bar Plot (수평 막대 그래프)
cfm = df[df['cut'] == 'Fair']['price'].mean()
plt.figure(figsize=(8,5))
sns.barplot(y='cut', x='price', hue='cut', data=df, palette='plasma', dodge=False)
plt.axvline(x=cfm, color='red', linestyle='--', linewidth=1.2, label='Fair Cut Mean Price')
plt.title("Mean Price by Cut", fontsize=16, fontweight='bold', pad=15)
plt.xlabel("Mean Price", fontsize=12)
plt.ylabel("Cut", fontsize=12)
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

# Line Chart (선 그래프)
bins = np.arange(df['carat'].min(), df['carat'].max() + 0.5, 0.5)
df['carat_binned'] = pd.cut(df['carat'], bins=bins)
avg = df.groupby('carat_binned')['price'].mean().reset_index().dropna()
avg['carat_center'] = avg['carat_binned'].apply(lambda x: x.mid)

fig = px.line(avg, x='carat_center', y='price', markers=True,
              title='Mean Price by Carat (0.5 Bins)',
              labels={'carat_center': 'Carat', 'price': 'Mean Price'})
fig.update_traces(hovertemplate='Carat: %{x:.2f}<br>Mean Price: %{y:.2f}',
                  line_color='royalblue', marker_size=8)
fig.update_layout(xaxis=dict(tickvals=avg['carat_center'], ticktext=[f"{x:.2f}" for x in avg['carat_center']]),
                  hovermode='x unified', template='plotly_white',
                  width=900, height=600)
fig.show()

# Area Plot (영역 그래프)
avg = df.groupby('carat')['price'].mean()
ax = avg.plot.area(figsize=(10, 6), alpha=0.5, color='teal', linewidth=2)
ax.set(title="Mean Price by Carat", xlabel="Carat", ylabel="Mean Price")
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# Histogram (히스토그램)
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='price', bins=40, color='#4DB6AC', edgecolor='white')
plt.title("Price Distribution", fontsize=16, fontweight='bold', pad=15)
plt.xlabel("Price", fontsize=13)
plt.ylabel("Frequency", fontsize=13)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.tight_layout()
plt.show()

# Pie Chart
cut_counts = df['cut'].value_counts()
colors = sns.color_palette('pastel')
plt.figure(figsize=(7,7))
plt.pie(
    cut_counts,
    labels=cut_counts.index,
    autopct='%1.1f%%',
    startangle=140,
    colors=colors,
    textprops={'fontsize': 12, 'color': 'dimgray'},
    wedgeprops={'edgecolor': 'white', 'linewidth': 1},
    shadow=True
)
plt.title("Diamond Cut Ratio", fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()

# Box Plot (박스 플롯)
plt.figure(figsize=(8,6))
sns.boxplot(x='cut', y='price', data=df, palette='viridis')
plt.title("Price Distribution by Cut")
plt.xlabel("Cut")
plt.ylabel("Price")
plt.show()

# 3D Scatter Plot using plotly (3D 산점도)
fig = px.scatter_3d(df, x='carat', y='price', z='cut',
                    color='cut', title='3D Scatterplot of Carat vs Price vs Cut',
                    labels={'carat': 'Carat', 'price': 'Price', 'cut': 'Cut'})
fig.show()

# Implot (회귀 플롯)
sns.lmplot(
    data=df, x='carat', y='price', hue='cut',
    height=6, palette='viridis',
    scatter_kws={'alpha':0.3, 's':20}
)

plt.title("Carat vs. Price Linear Regression by Cut", fontsize=16, fontweight='bold')
plt.xlabel("Carat", fontsize=14)
plt.ylabel("Price", fontsize=14)
plt.tight_layout()
plt.show()

# Countplot (빈도 그래프)
plt.figure(figsize=(8,5))
sns.countplot(x='cut', data=df, palette='plasma')
plt.title("Diamond Count by Cut")
plt.xlabel("Cut")
plt.ylabel("Count")
plt.show()

# Heatmap (히트맵)
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Joint Plot (조인트 플롯)
sample_df = df.sample(n=2000, random_state=42)  # 2000개 샘플링

jp = sns.jointplot(
    data=sample_df,
    x='carat',
    y='price',
    kind='reg',
    color='green',
    scatter_kws={'s':20, 'alpha':0.5}
)
jp.set_axis_labels("Carat", "Price", fontsize=14)
plt.suptitle("\nCarat vs Price with Regression Line", fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# KDE Plot (커널 밀도 추정)
plt.figure(figsize=(8,6))
sns.kdeplot(df['carat'], fill=True, color='purple')
plt.title("Carat Kernel Density Estimation")
plt.xlabel("Carat")
plt.ylabel("Density")
plt.show()

# Pairplot (쌍별 플롯)
sample_df = df.sample(n=500, random_state=42)
sns.pairplot(sample_df[['carat', 'depth', 'price', 'table', 'cut']], hue='cut', palette='tab10')
plt.show()

# relplot으로 산점도 그리기
sns.relplot(
    data=df,
    x='carat',
    y='price',
    kind='scatter',
    hue='cut',
    height=8,
    aspect=1.2,
    palette='muted',
    alpha=0.3,
)

plt.title("Carat vs Price by Cut", fontsize=18, fontweight='bold', pad=15)
plt.xlabel("Carat", fontsize=14, labelpad=15)
plt.ylabel("Price", fontsize=14, labelpad=15)

plt.grid(True, linestyle='--', alpha=0.4)
plt.tick_params(axis='both', which='major', labelsize=12, colors='dimgray')

plt.tight_layout()
plt.show()

# Rugplot (러그 플롯)
plt.figure(figsize=(12, 2))
sns.rugplot(df['carat'], color='coral', height=0.1, linewidth=1.5)
plt.title("Carat Rug Plot", fontsize=16)
plt.xlabel("Carat", fontsize=14)
plt.yticks([])
plt.tight_layout()
plt.show()

# Violin Plot (바이올린 플롯)
plt.figure(figsize=(8,6))
sns.violinplot(x='cut', y='price', data=df, hue='cut', palette='viridis', legend=False)
plt.title("Price Distribution by Cut")
plt.xlabel("Cut")
plt.ylabel("Price")
plt.show()

# Interactive Plot (plotly 산점도)
fig = px.scatter(df, x='carat', y='price', color='cut', title='Interactive Scatterplot',
                 labels={'carat': 'Carat', 'price': 'Price'})
fig.show()