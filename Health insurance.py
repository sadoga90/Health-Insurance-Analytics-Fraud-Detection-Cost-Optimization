#!/usr/bin/env python
# coding: utf-8

# In[10]:


import matplotlib.pyplot as plt
from arabic_reshaper import reshape
from bidi.algorithm import get_display

plt.rcParams['font.family'] = 'Arial'   # أو Cairo أو Amiri
plt.rcParams['axes.unicode_minus'] = False

def ar(text):
    return get_display(reshape(text))


# In[11]:


# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Load the dataset
df = pd.read_csv('insurance.csv')

# Display basic information
print("="*60)
print("المعلومات الأساسية عن مجموعة البيانات")
print("="*60)
print(f"عدد الصفوف: {df.shape[0]}")
print(f"عدد الأعمدة: {df.shape[1]}")
print("\nالأعمدة المتاحة:")
print(df.columns.tolist())


# In[12]:


# Display first few rows
print("\n" + "="*60)
print("عرض أول 10 صفوف من البيانات")
print("="*60)
print(df.head(10))

# Display last few rows
print("\n" + "="*60)
print("عرض آخر 10 صفوف من البيانات")
print("="*60)
print(df.tail(10))


# In[13]:


# Check data types and missing values
print("\n" + "="*60)
print("معلومات عن أنواع البيانات والقيم المفقودة")
print("="*60)
print(df.info())

print("\n" + "="*60)
print("التحقق من القيم المفقودة")
print("="*60)
print(df.isnull().sum())


# In[14]:


# Descriptive statistics for numerical variables
print("\n" + "="*60)
print("الإحصائيات الوصفية للمتغيرات العددية")
print("="*60)
print(df.describe().round(2))

# Additional statistics
print("\n" + "="*60)
print("إحصائيات إضافية")
print("="*60)
print(f"الانحراف المعياري للتكاليف: {df['charges'].std():.2f}")
print(f"معامل الاختلاف للتكاليف: {(df['charges'].std() / df['charges'].mean() * 100):.2f}%")
print(f"نطاق التكاليف (Range): {df['charges'].max() - df['charges'].min():.2f}")
print(f"النسبة بين Q3 وQ1 للتكاليف: {df['charges'].quantile(0.75) - df['charges'].quantile(0.25):.2f}")


# In[15]:


# Descriptive statistics for categorical variables
categorical_cols = ['sex', 'smoker', 'region', 'children']

print("\n" + "="*60)
print("الإحصائيات الوصفية للمتغيرات الفئوية")
print("="*60)

for col in categorical_cols:
    print(f"\n{col.upper()}:")
    print(df[col].value_counts())
    print(f"النسبة المئوية:")
    print((df[col].value_counts(normalize=True) * 100).round(2))
    print("-"*40)


# In[16]:


# Check for outliers using IQR method
print("\n" + "="*60)
print("التحقق من القيم المتطرفة باستخدام طريقة IQR")
print("="*60)

numerical_cols = ['age', 'bmi', 'charges']
for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    print(f"\n{col.upper()}:")
    print(f"عدد القيم المتطرفة: {len(outliers)}")
    print(f"نسبة القيم المتطرفة: {(len(outliers)/len(df)*100):.2f}%")
    print(f"الحد الأدنى: {lower_bound:.2f}, الحد الأعلى: {upper_bound:.2f}")


# In[26]:


get_ipython().system('pip install arabic-reshaper python-bidi')

from arabic_reshaper import reshape
from bidi.algorithm import get_display

def fix_arabic(text):
    return get_display(reshape(text))


# In[27]:


fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle(fix_arabic('توزيع المتغيرات العددية'), fontsize=16, fontweight='bold')

# Age distribution
axes[0, 0].hist(df['age'], bins=20, edgecolor='black', alpha=0.7)
axes[0, 0].axvline(df['age'].mean(), color='red', linestyle='--', 
                   label=fix_arabic(f'المتوسط: {df["age"].mean():.1f}'))
axes[0, 0].axvline(df['age'].median(), color='green', linestyle='--', 
                   label=fix_arabic(f'الوسيط: {df["age"].median():.1f}'))
axes[0, 0].set_title(fix_arabic('توزيع العمر'))
axes[0, 0].set_xlabel(fix_arabic('العمر'))
axes[0, 0].set_ylabel(fix_arabic('التكرار'))
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# BMI distribution
axes[0, 1].hist(df['bmi'], bins=20, edgecolor='black', alpha=0.7, color='orange')
axes[0, 1].axvline(df['bmi'].mean(), color='red', linestyle='--', 
                   label=fix_arabic(f'المتوسط: {df["bmi"].mean():.1f}'))
axes[0, 1].axvline(df['bmi'].median(), color='green', linestyle='--', 
                   label=fix_arabic(f'الوسيط: {df["bmi"].median():.1f}'))
axes[0, 1].axvline(18.5, color='blue', linestyle=':', label=fix_arabic('نقص الوزن'))
axes[0, 1].axvline(25, color='blue', linestyle=':', label=fix_arabic('وزن طبيعي'))
axes[0, 1].axvline(30, color='blue', linestyle=':', label=fix_arabic('زيادة الوزن'))
axes[0, 1].set_title(fix_arabic('توزيع مؤشر كتلة الجسم (BMI)'))
axes[0, 1].set_xlabel(fix_arabic('مؤشر كتلة الجسم'))
axes[0, 1].set_ylabel(fix_arabic('التكرار'))
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Charges distribution
axes[1, 0].hist(df['charges'], bins=30, edgecolor='black', alpha=0.7, color='green')
axes[1, 0].axvline(df['charges'].mean(), color='red', linestyle='--', 
                   label=fix_arabic(f'المتوسط: {df["charges"].mean():.0f}'))
axes[1, 0].axvline(df['charges'].median(), color='green', linestyle='--', 
                   label=fix_arabic(f'الوسيط: {df["charges"].median():.0f}'))
axes[1, 0].set_title(fix_arabic('توزيع التكاليف'))
axes[1, 0].set_xlabel(fix_arabic('التكاليف ($)'))
axes[1, 0].set_ylabel(fix_arabic('التكرار'))
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Children distribution
children_counts = df['children'].value_counts().sort_index()
axes[1, 1].bar(children_counts.index, children_counts.values, edgecolor='black', alpha=0.7)
axes[1, 1].set_title(fix_arabic('توزيع عدد الأطفال'))
axes[1, 1].set_xlabel(fix_arabic('عدد الأطفال'))
axes[1, 1].set_ylabel(fix_arabic('التكرار'))
for i, v in enumerate(children_counts.values):
    axes[1, 1].text(i, v + 10, str(v), ha='center', fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# In[ ]:


# Create visualizations for categorical variables

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle(ar('توزيع المتغيرات الفئوية'), fontsize=16, fontweight='bold')

# Gender distribution
gender_counts = df['sex'].value_counts()
axes[0, 0].pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%', 
                colors=['lightblue', 'lightpink'], startangle=90)
axes[0, 0].set_title(ar('توزيع الجنس'))
axes[0, 0].legend(title=ar('الجنس'), loc='upper right')

# Smoker distribution
smoker_counts = df['smoker'].value_counts()
axes[0, 1].bar(smoker_counts.index, smoker_counts.values, color=['lightgreen', 'salmon'], edgecolor='black')
axes[0, 1].set_title(ar('توزيع المدخنين'))
axes[0, 1].set_xlabel(ar('حالة التدخين'))
axes[0, 1].set_ylabel(ar('عدد الأشخاص'))
for i, v in enumerate(smoker_counts.values):
    axes[0, 1].text(i, v + 10, str(v), ha='center', fontweight='bold')

# Region distribution
region_counts = df['region'].value_counts()
axes[1, 0].bar(region_counts.index, region_counts.values, color='skyblue', edgecolor='black')
axes[1, 0].set_title(ar('توزيع المناطق'))
axes[1, 0].set_xlabel(ar('المنطقة'))
axes[1, 0].set_ylabel(ar('عدد الأشخاص'))
axes[1, 0].tick_params(axis='x', rotation=45)
for i, v in enumerate(region_counts.values):
    axes[1, 0].text(i, v + 10, str(v), ha='center', fontweight='bold')

# Children as categorical
children_cat_counts = df['children'].value_counts().sort_index()
axes[1, 1].bar(children_cat_counts.index.astype(str), children_cat_counts.values, color='lightcoral', edgecolor='black')
axes[1, 1].set_title(ar('توزيع عدد الأطفال (كمتغير فئوي)'))
axes[1, 1].set_xlabel(ar('عدد الأطفال'))
axes[1, 1].set_ylabel(ar('عدد الأشخاص'))

for i, v in enumerate(children_cat_counts.values):
    axes[1, 1].text(i, v + 10, str(v), ha='center', fontweight='bold')

plt.tight_layout()
plt.show()


# In[ ]:


# Correlation matrix and pairplot
print("\n" + "="*60)
print("مصفوفة الارتباط بين المتغيرات العددية")
print("="*60)

# Calculate correlation matrix
correlation_matrix = df[['age', 'bmi', 'children', 'charges']].corr()
print(correlation_matrix.round(3))

# Visualize correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title(ar('مصفوفة الارتباط بين المتغيرات'), fontsize=16, fontweight='bold')
plt.show()


# In[18]:


# Create pairplot with hue for smoker status
print("\n" + "="*60)
print("العلاقات بين المتغيرات الرئيسية")
print("="*60)

# Select subset for pairplot
pairplot_vars = df[['age', 'bmi', 'charges', 'smoker', 'sex']]
g = sns.pairplot(pairplot_vars, hue='smoker', palette={'yes': 'red', 'no': 'green'}, 
                  diag_kind='kde', height=3, aspect=1.2)
g.fig.suptitle(ar('العلاقات بين العمر، مؤشر كتلة الجسم، والتكاليف حسب حالة التدخين'), 
                y=1.02, fontsize=14, fontweight='bold')
plt.show()


# In[19]:


# Analyze charges by different categories
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle(ar('تحليل التكاليف حسب الفئات المختلفة'), fontsize=16, fontweight='bold')

# Charges by smoking status
sns.boxplot(data=df, x='smoker', y='charges', ax=axes[0, 0], palette={'yes': 'lightcoral', 'no': 'lightgreen'})
axes[0, 0].set_title(ar('التكاليف حسب حالة التدخين'))
axes[0, 0].set_xlabel(ar('مدخن'))
axes[0, 0].set_ylabel(ar('التكاليف ($)'))

# Charges by gender
sns.boxplot(data=df, x='sex', y='charges', ax=axes[0, 1], palette={'male': 'lightblue', 'female': 'lightpink'})
axes[0, 1].set_title(ar('التكاليف حسب الجنس'))
axes[0, 1].set_xlabel(ar('الجنس'))
axes[0, 1].set_ylabel(ar('التكاليف ($)'))

# Charges by region
sns.boxplot(data=df, x='region', y='charges', ax=axes[1, 0], palette='Set2')
axes[1, 0].set_title(ar('التكاليف حسب المنطقة'))
axes[1, 0].set_xlabel(ar('المنطقة'))
axes[1, 0].set_ylabel(ar('التكاليف ($)'))
axes[1, 0].tick_params(axis='x', rotation=45)

# Charges by number of children
sns.boxplot(data=df, x='children', y='charges', ax=axes[1, 1], palette='viridis')
axes[1, 1].set_title(ar('التكاليف حسب عدد الأطفال'))
axes[1, 1].set_xlabel(ar('عدد الأطفال'))
axes[1, 1].set_ylabel(ar('التكاليف ($)'))

plt.tight_layout()
plt.show()


# In[28]:


correlation_matrix = df.corr(numeric_only=True)


# In[29]:


# Create final summary report
print("\n" + "="*80)
print("تقرير تحليل البيانات الأولي - التأمين الصحي")
print("="*80)

print("\n📊 ملخص البيانات:")
print(f"   • إجمالي السجلات: {df.shape[0]}")
print(f"   • عدد المتغيرات: {df.shape[1]}")
print(f"   • لا توجد قيم مفقودة في البيانات")

print("\n🔢 المتغيرات العددية:")
print(f"   • العمر: من {df['age'].min()} إلى {df['age'].max()} سنة (متوسط: {df['age'].mean():.1f})")
print(f"   • مؤشر كتلة الجسم: من {df['bmi'].min():.1f} إلى {df['bmi'].max():.1f} (متوسط: {df['bmi'].mean():.1f})")
print(f"   • التكاليف: من ${df['charges'].min():.2f} إلى ${df['charges'].max():.2f}")
print(f"   • متوسط التكاليف: ${df['charges'].mean():.2f} (وسيط: ${df['charges'].median():.2f})")

print("\n🏷️ المتغيرات الفئوية:")
print(f"   • الجنس: {df['sex'].value_counts()['male']} ذكر ({df['sex'].value_counts(normalize=True)['male']*100:.1f}%) و {df['sex'].value_counts()['female']} أنثى ({df['sex'].value_counts(normalize=True)['female']*100:.1f}%)")
print(f"   • المدخنون: {df['smoker'].value_counts()['yes']} شخص ({df['smoker'].value_counts(normalize=True)['yes']*100:.1f}%)")
print(f"   • المناطق: {', '.join([f'{region} ({count})' for region, count in df['region'].value_counts().items()])}")
print(f"   • عدد الأطفال: {df['children'].value_counts().sort_index().to_dict()}")

print("\n📈 الارتباطات المهمة:")
print(f"   • أقوى ارتباط إيجابي: العمر مع التكاليف ({correlation_matrix.loc['age', 'charges']:.3f})")
print(f"   • ارتباط مؤشر كتلة الجسم مع التكاليف: {correlation_matrix.loc['bmi', 'charges']:.3f}")
print(f"   • ارتباط عدد الأطفال مع التكاليف: {correlation_matrix.loc['children', 'charges']:.3f}")

print("\n💡 ملاحظات أولية:")
print("   1. بيانات التكاليف منحرفة لليمين بشدة (توجد قيم متطرفة عالية)")
print("   2. المدخنون لديهم تكاليف أعلى بشكل ملحوظ")
print("   3. التوزيع العمري متوازن بشكل جيد")
print("   4. معظم الأفراد لديهم وزن زائد (متوسط BMI = 30.66)")
print("   5. لا توجد فروق كبيرة في التكاليف بين المناطق")

print("\n" + "="*80)

