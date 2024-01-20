# Importing the Libraries
import numpy as np
import pandas as pd
import datetime
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import AgglomerativeClustering
from matplotlib.colors import ListedColormap
from sklearn import metrics
import warnings
import distutils
import sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")
np.random.seed(42)

def remove_na_values(df):
    # To remove the NA values
    data = df.dropna()
    print("The total number of data-points after removing the rows with missing values are:", len(data))
    return data

def convert_date_column(data):
    # Convert the "Dt_Customer" column to datetime
    data["Dt_Customer"] = pd.to_datetime(data["Dt_Customer"])
    dates = [i.date() for i in data["Dt_Customer"]]

    print("The newest customer's enrollment date in the records:", max(dates))
    print("The oldest customer's enrollment date in the records:", min(dates))

    # Created a feature "Customer_For"
    days = [(max(dates) - i).days for i in dates]
    data["Customer_For"] = days
    data["Customer_For"] = pd.to_numeric(data["Customer_For"], errors="coerce")

    return data

def feature_engineering(data):
    # Feature Engineering
    data["Age"] = 2021 - data["Year_Birth"]
    data["Spent"] = data[["MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds"]].sum(axis=1)
    data["Living_With"] = data["Marital_Status"].replace({"Married": "Partner", "Together": "Partner", "Absurd": "Alone", "Widow": "Alone", "YOLO": "Alone", "Divorced": "Alone", "Single": "Alone", })
    data["Children"] = data["Kidhome"] + data["Teenhome"]
    data["Family_Size"] = data["Living_With"].replace({"Alone": 1, "Partner": 2}) + data["Children"]
    data["Is_Parent"] = np.where(data.Children > 0, 1, 0)
    data["Education"] = data["Education"].replace({"Basic": "Undergraduate", "2n Cycle": "Undergraduate", "Graduation": "Graduate", "Master": "Postgraduate", "PhD": "Postgraduate"})
    data = data.rename(columns={"MntWines": "Wines", "MntFruits": "Fruits", "MntMeatProducts": "Meat", "MntFishProducts": "Fish", "MntSweetProducts": "Sweets", "MntGoldProds": "Gold"})

    # Dropping some of the redundant features
    to_drop = ["Marital_Status", "Dt_Customer", "Z_CostContact", "Z_Revenue", "Year_Birth", "ID"]
    data = data.drop(to_drop, axis=1)

    return data

def relative_plot(data):
    # Relative Plot Of Some Selected Features: A Data Subset
    To_Plot = ["Income", "Recency", "Customer_For", "Age", "Spent", "Is_Parent"]
    sns.set(rc={"axes.facecolor": "#FFF9ED", "figure.facecolor": "#FFF9ED"})
    pallet = ["#682F2F", "#9E726F", "#D6B2B1", "#B9C0C9", "#9F8A78", "#F3AB60"]
    cmap = colors.ListedColormap(["#682F2F", "#9E726F", "#D6B2B1", "#B9C0C9", "#9F8A78", "#F3AB60"])

    plt.figure()
    sns.pairplot(data[To_Plot], hue="Is_Parent", palette=(["#682F2F", "#F3AB60"]))
    plt.show()

def remove_outliers(data):
    # Dropping the outliers by setting a cap on Age and income.
    data = data[(data["Age"] < 90)]
    data = data[(data["Income"] < 600000)]
    print("The total number of data-points after removing the outliers are:", len(data))
    return data

def label_encoding(data, object_cols):
    # Label Encoding the object dtypes.
    LE = LabelEncoder()
    for i in object_cols:
        data[i] = data[[i]].apply(LE.fit_transform)
    print("All features are now numerical")
    return data

def create_scaled_dataset(data):
    # Creating a copy of data
    ds = data.copy()
    # Creating a subset of the dataframe by dropping the features on deals accepted and promotions
    cols_del = ['AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1', 'AcceptedCmp2', 'Complain', 'Response']
    ds = ds.drop(cols_del, axis=1)
    # Scaling
    scaler = StandardScaler()
    scaler.fit(ds)
    scaled_ds = pd.DataFrame(scaler.transform(ds), columns=ds.columns)
    print("All features are now scaled")
    return scaled_ds

def apply_pca(scaled_ds):
    # Initiating PCA to reduce dimensions aka features to 3
    pca = PCA(n_components=3)
    pca.fit(scaled_ds)
    PCA_ds = pd.DataFrame(pca.transform(scaled_ds), columns=(["col1", "col2", "col3"]))
    return PCA_ds

def plot_3d_projection(PCA_ds):
    # A 3D Projection Of Data In The Reduced Dimension
    x = PCA_ds["col1"]
    y = PCA_ds["col2"]
    z = PCA_ds["col3"]
    # To plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x, y, z, c="maroon", marker="o")
    ax.set_title("A 3D Projection Of Data In The Reduced Dimension")
    plt.show()

def elbow_method(scaled_ds):
    # Quick examination of the elbow method to find the numbers of clusters to make.
    print('Elbow Method to determine the number of clusters to be formed:')
    Elbow_M = KElbowVisualizer(KMeans(), k=10)
    Elbow_M.fit(scaled_ds)
    Elbow_M.show()

def plot_clusters_3d(PCA_ds):
    # Plotting the clusters
    x = PCA_ds["col1"]
    y = PCA_ds["col2"]
    z = PCA_ds["col3"]
    fig = plt.figure(figsize=(10, 8))
    ax = plt.subplot(111, projection='3d', label="bla")
    ax.scatter(x, y, z, s=40, c=PCA_ds["Clusters"], marker='o', cmap=cmap)
    ax.set_title("The Plot Of The Clusters")
    plt.show()

def plot_clusters_distribution(data):
    # Plotting count plot of clusters
    pal = ["#682F2F", "#B9C0C9", "#9F8A78", "#F3AB60"]
    pl = sns.countplot(x=data["Clusters"], palette=pal)
    pl.set_title("Distribution Of The Clusters")
    plt.show()

def plot_cluster_profile(data):
    # Plotting cluster's profile based on Income And Spending
    pl = sns.scatterplot(data=data, x=data["Spent"], y=data["Income"], hue=data["Clusters"], palette=pal)
    pl.set_title("Cluster's Profile Based On Income And Spending")
    plt.legend()
    plt.show()

def plot_spending_distribution(data):
    # Plotting spending distribution
    plt.figure()
    pl = sns.swarmplot(x=data["Clusters"], y=data["Spent"], color="#CBEDDD", alpha=0.5)
    pl = sns.boxenplot(x=data["Clusters"], y=data["Spent"], palette=pal)
    plt.show()

def create_total_promos_feature(data):
    # Creating a feature to get a sum of accepted promotions
    data["Total_Promos"] = data["AcceptedCmp1"] + data["AcceptedCmp2"] + data["AcceptedCmp3"] + data["AcceptedCmp4"] + data["AcceptedCmp5"]

def plot_promotions_count(data):
    # Plotting count of total campaign accepted.
    plt.figure()
    pl = sns.countplot(x=data["Total_Promos"], hue=data["Clusters"], palette=pal)
    pl.set_title("Count Of Promotion Accepted")
    pl.set_xlabel("Number Of Total Accepted Promotions")
    plt.show()

def plot_deals_purchased(data):
    # Plotting the number of deals purchased
    plt.figure()
    pl = sns.boxenplot(y=data["NumDealsPurchases"], x=data["Clusters"], palette=pal)
    pl.set_title("Number of Deals Purchased")
    plt.show()

def main():
    # Example usage
    df = pd.read_csv(r"E:\ZK00_P2\marketing_campaign.csv")  # Load your data
    data = remove_na_values(df)
    data = convert_date_column(data)
    data = feature_engineering(data)
    relative_plot(data)
    data = remove_outliers(data)
    object_cols = ["Your_Object_Columns", "Go_Here"]
    data = label_encoding(data, object_cols)
    scaled_ds = create_scaled_dataset(data)
    PCA_ds = apply_pca(scaled_ds)
    plot_3d_projection(PCA_ds)
    elbow_method(scaled_ds)
    plot_clusters_3d(PCA_ds)
    plot_clusters_distribution(data)
    plot_cluster_profile(data)
    plot_spending_distribution(data)
    create_total_promos_feature(data)
    plot_promotions_count(data)
    plot_deals_purchased(data)

if __name__ == "__main__":
    main()