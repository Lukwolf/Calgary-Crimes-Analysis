{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Data Preparation and Cleaning"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sn\n",
    "import geopandas as gpd\n",
    "from shapely.geometry import Point"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Import the Dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Sector</th>\n",
       "      <th>Community Name</th>\n",
       "      <th>Category</th>\n",
       "      <th>Crime Count</th>\n",
       "      <th>Resident Count</th>\n",
       "      <th>Date</th>\n",
       "      <th>Year</th>\n",
       "      <th>Month</th>\n",
       "      <th>ID</th>\n",
       "      <th>Community Center Point</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>NORTHWEST</td>\n",
       "      <td>ARBOUR LAKE</td>\n",
       "      <td>Theft OF Vehicle</td>\n",
       "      <td>2</td>\n",
       "      <td>10619.0</td>\n",
       "      <td>2022/04</td>\n",
       "      <td>2022</td>\n",
       "      <td>APR</td>\n",
       "      <td>2022-APR-ARBOUR LAKE-Theft OF Vehicle</td>\n",
       "      <td>POINT (-114.20767498075155 51.1325947114686)</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>CENTRE</td>\n",
       "      <td>BANFF TRAIL</td>\n",
       "      <td>Theft OF Vehicle</td>\n",
       "      <td>2</td>\n",
       "      <td>4153.0</td>\n",
       "      <td>2023/10</td>\n",
       "      <td>2023</td>\n",
       "      <td>OCT</td>\n",
       "      <td>2023-OCT-BANFF TRAIL-Theft OF Vehicle</td>\n",
       "      <td>POINT (-114.11512839716917 51.07421633024228)</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>EAST</td>\n",
       "      <td>DOVER</td>\n",
       "      <td>Theft OF Vehicle</td>\n",
       "      <td>5</td>\n",
       "      <td>10351.0</td>\n",
       "      <td>2022/12</td>\n",
       "      <td>2022</td>\n",
       "      <td>DEC</td>\n",
       "      <td>2022-DEC-DOVER-Theft OF Vehicle</td>\n",
       "      <td>POINT (-113.99305400906283 51.02256772250409)</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>CENTRE</td>\n",
       "      <td>GREENVIEW</td>\n",
       "      <td>Assault (Non-domestic)</td>\n",
       "      <td>2</td>\n",
       "      <td>1906.0</td>\n",
       "      <td>2020/12</td>\n",
       "      <td>2020</td>\n",
       "      <td>DEC</td>\n",
       "      <td>2020-DEC-GREENVIEW-Assault (Non-domestic)</td>\n",
       "      <td>POINT (-114.05746990262463 51.09485613506574)</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>NORTHWEST</td>\n",
       "      <td>HAMPTONS</td>\n",
       "      <td>Theft FROM Vehicle</td>\n",
       "      <td>3</td>\n",
       "      <td>7382.0</td>\n",
       "      <td>2019/08</td>\n",
       "      <td>2019</td>\n",
       "      <td>AUG</td>\n",
       "      <td>2019-AUG-HAMPTONS-Theft FROM Vehicle</td>\n",
       "      <td>POINT (-114.14668419231347 51.14509283969437)</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "      Sector Community Name                Category  Crime Count  \\\n",
       "0  NORTHWEST    ARBOUR LAKE        Theft OF Vehicle            2   \n",
       "1     CENTRE    BANFF TRAIL        Theft OF Vehicle            2   \n",
       "2       EAST          DOVER        Theft OF Vehicle            5   \n",
       "3     CENTRE      GREENVIEW  Assault (Non-domestic)            2   \n",
       "4  NORTHWEST       HAMPTONS      Theft FROM Vehicle            3   \n",
       "\n",
       "   Resident Count     Date  Year Month  \\\n",
       "0         10619.0  2022/04  2022   APR   \n",
       "1          4153.0  2023/10  2023   OCT   \n",
       "2         10351.0  2022/12  2022   DEC   \n",
       "3          1906.0  2020/12  2020   DEC   \n",
       "4          7382.0  2019/08  2019   AUG   \n",
       "\n",
       "                                          ID  \\\n",
       "0      2022-APR-ARBOUR LAKE-Theft OF Vehicle   \n",
       "1      2023-OCT-BANFF TRAIL-Theft OF Vehicle   \n",
       "2            2022-DEC-DOVER-Theft OF Vehicle   \n",
       "3  2020-DEC-GREENVIEW-Assault (Non-domestic)   \n",
       "4       2019-AUG-HAMPTONS-Theft FROM Vehicle   \n",
       "\n",
       "                          Community Center Point  \n",
       "0   POINT (-114.20767498075155 51.1325947114686)  \n",
       "1  POINT (-114.11512839716917 51.07421633024228)  \n",
       "2  POINT (-113.99305400906283 51.02256772250409)  \n",
       "3  POINT (-114.05746990262463 51.09485613506574)  \n",
       "4  POINT (-114.14668419231347 51.14509283969437)  "
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "df = pd.read_csv(r\"Community_Crime_Statistics_20240130.csv\")\n",
    "display(df.head())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Check for types"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Sector                     object\n",
       "Community Name             object\n",
       "Category                   object\n",
       "Crime Count                 int64\n",
       "Resident Count            float64\n",
       "Date                       object\n",
       "Year                        int64\n",
       "Month                      object\n",
       "ID                         object\n",
       "Community Center Point     object\n",
       "dtype: object"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.dtypes"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Transform 'Community Center Point' into 3 other columns for geospacial plots"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Sector</th>\n",
       "      <th>Community Name</th>\n",
       "      <th>Category</th>\n",
       "      <th>Crime Count</th>\n",
       "      <th>Resident Count</th>\n",
       "      <th>Date</th>\n",
       "      <th>Year</th>\n",
       "      <th>Month</th>\n",
       "      <th>ID</th>\n",
       "      <th>Community Center Point</th>\n",
       "      <th>Longitude</th>\n",
       "      <th>Latitude</th>\n",
       "      <th>geometry</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>NORTHWEST</td>\n",
       "      <td>ARBOUR LAKE</td>\n",
       "      <td>Theft OF Vehicle</td>\n",
       "      <td>2</td>\n",
       "      <td>10619.0</td>\n",
       "      <td>2022/04</td>\n",
       "      <td>2022</td>\n",
       "      <td>APR</td>\n",
       "      <td>2022-APR-ARBOUR LAKE-Theft OF Vehicle</td>\n",
       "      <td>POINT (-114.20767498075155 51.1325947114686)</td>\n",
       "      <td>-114.207675</td>\n",
       "      <td>51.132595</td>\n",
       "      <td>POINT (-114.20767 51.13259)</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>CENTRE</td>\n",
       "      <td>BANFF TRAIL</td>\n",
       "      <td>Theft OF Vehicle</td>\n",
       "      <td>2</td>\n",
       "      <td>4153.0</td>\n",
       "      <td>2023/10</td>\n",
       "      <td>2023</td>\n",
       "      <td>OCT</td>\n",
       "      <td>2023-OCT-BANFF TRAIL-Theft OF Vehicle</td>\n",
       "      <td>POINT (-114.11512839716917 51.07421633024228)</td>\n",
       "      <td>-114.115128</td>\n",
       "      <td>51.074216</td>\n",
       "      <td>POINT (-114.11513 51.07422)</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>EAST</td>\n",
       "      <td>DOVER</td>\n",
       "      <td>Theft OF Vehicle</td>\n",
       "      <td>5</td>\n",
       "      <td>10351.0</td>\n",
       "      <td>2022/12</td>\n",
       "      <td>2022</td>\n",
       "      <td>DEC</td>\n",
       "      <td>2022-DEC-DOVER-Theft OF Vehicle</td>\n",
       "      <td>POINT (-113.99305400906283 51.02256772250409)</td>\n",
       "      <td>-113.993054</td>\n",
       "      <td>51.022568</td>\n",
       "      <td>POINT (-113.99305 51.02257)</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>CENTRE</td>\n",
       "      <td>GREENVIEW</td>\n",
       "      <td>Assault (Non-domestic)</td>\n",
       "      <td>2</td>\n",
       "      <td>1906.0</td>\n",
       "      <td>2020/12</td>\n",
       "      <td>2020</td>\n",
       "      <td>DEC</td>\n",
       "      <td>2020-DEC-GREENVIEW-Assault (Non-domestic)</td>\n",
       "      <td>POINT (-114.05746990262463 51.09485613506574)</td>\n",
       "      <td>-114.057470</td>\n",
       "      <td>51.094856</td>\n",
       "      <td>POINT (-114.05747 51.09486)</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>NORTHWEST</td>\n",
       "      <td>HAMPTONS</td>\n",
       "      <td>Theft FROM Vehicle</td>\n",
       "      <td>3</td>\n",
       "      <td>7382.0</td>\n",
       "      <td>2019/08</td>\n",
       "      <td>2019</td>\n",
       "      <td>AUG</td>\n",
       "      <td>2019-AUG-HAMPTONS-Theft FROM Vehicle</td>\n",
       "      <td>POINT (-114.14668419231347 51.14509283969437)</td>\n",
       "      <td>-114.146684</td>\n",
       "      <td>51.145093</td>\n",
       "      <td>POINT (-114.14668 51.14509)</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "      Sector Community Name                Category  Crime Count  \\\n",
       "0  NORTHWEST    ARBOUR LAKE        Theft OF Vehicle            2   \n",
       "1     CENTRE    BANFF TRAIL        Theft OF Vehicle            2   \n",
       "2       EAST          DOVER        Theft OF Vehicle            5   \n",
       "3     CENTRE      GREENVIEW  Assault (Non-domestic)            2   \n",
       "4  NORTHWEST       HAMPTONS      Theft FROM Vehicle            3   \n",
       "\n",
       "   Resident Count     Date  Year Month  \\\n",
       "0         10619.0  2022/04  2022   APR   \n",
       "1          4153.0  2023/10  2023   OCT   \n",
       "2         10351.0  2022/12  2022   DEC   \n",
       "3          1906.0  2020/12  2020   DEC   \n",
       "4          7382.0  2019/08  2019   AUG   \n",
       "\n",
       "                                          ID  \\\n",
       "0      2022-APR-ARBOUR LAKE-Theft OF Vehicle   \n",
       "1      2023-OCT-BANFF TRAIL-Theft OF Vehicle   \n",
       "2            2022-DEC-DOVER-Theft OF Vehicle   \n",
       "3  2020-DEC-GREENVIEW-Assault (Non-domestic)   \n",
       "4       2019-AUG-HAMPTONS-Theft FROM Vehicle   \n",
       "\n",
       "                          Community Center Point   Longitude   Latitude  \\\n",
       "0   POINT (-114.20767498075155 51.1325947114686) -114.207675  51.132595   \n",
       "1  POINT (-114.11512839716917 51.07421633024228) -114.115128  51.074216   \n",
       "2  POINT (-113.99305400906283 51.02256772250409) -113.993054  51.022568   \n",
       "3  POINT (-114.05746990262463 51.09485613506574) -114.057470  51.094856   \n",
       "4  POINT (-114.14668419231347 51.14509283969437) -114.146684  51.145093   \n",
       "\n",
       "                      geometry  \n",
       "0  POINT (-114.20767 51.13259)  \n",
       "1  POINT (-114.11513 51.07422)  \n",
       "2  POINT (-113.99305 51.02257)  \n",
       "3  POINT (-114.05747 51.09486)  \n",
       "4  POINT (-114.14668 51.14509)  "
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['Community Center Point'].apply(type).value_counts()\n",
    "df['Longitude'] = df['Community Center Point'].apply(lambda x: float(x.split()[1].lstrip('(')) if isinstance(x, str) else None)\n",
    "df['Latitude'] = df['Community Center Point'].apply(lambda x: float(x.split()[2].rstrip(')')) if isinstance(x, str) else None)\n",
    "gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Longitude, df.Latitude))\n",
    "gdf.crs = \"EPSG:4326\"\n",
    "gdf.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Check for columns types"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Sector                     object\n",
       "Community Name             object\n",
       "Category                   object\n",
       "Crime Count                 int64\n",
       "Resident Count            float64\n",
       "Date                       object\n",
       "Year                        int64\n",
       "Month                      object\n",
       "ID                         object\n",
       "Community Center Point     object\n",
       "Longitude                 float64\n",
       "Latitude                  float64\n",
       "dtype: object"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.dtypes"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Check for nan values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Sector                    31\n",
       "Community Name             0\n",
       "Category                   0\n",
       "Crime Count                0\n",
       "Resident Count            69\n",
       "Date                       0\n",
       "Year                       0\n",
       "Month                      0\n",
       "ID                         0\n",
       "Community Center Point    31\n",
       "Longitude                 31\n",
       "Latitude                  31\n",
       "dtype: int64"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.isna().sum()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Data Shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(67262, 12)"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "since the data has 67.262 rows and the max nan rows is 69, I am going to delete the NAN values, since they only represent 0.1% of the data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_clean = df.dropna(subset=['Resident Count'])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Now I am going to check for NaN values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Sector                    0\n",
       "Community Name            0\n",
       "Category                  0\n",
       "Crime Count               0\n",
       "Resident Count            0\n",
       "Date                      0\n",
       "Year                      0\n",
       "Month                     0\n",
       "ID                        0\n",
       "Community Center Point    0\n",
       "Longitude                 0\n",
       "Latitude                  0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_clean.isna().sum()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## The next step is transform 'Date' column into a date-time type"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\lukas\\AppData\\Local\\Temp\\ipykernel_19352\\353726464.py:1: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame.\n",
      "Try using .loc[row_indexer,col_indexer] = value instead\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  df_clean['Date'] = pd.to_datetime(df_clean['Date'], format='%Y/%m')\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "Sector                            object\n",
       "Community Name                    object\n",
       "Category                          object\n",
       "Crime Count                        int64\n",
       "Resident Count                   float64\n",
       "Date                      datetime64[ns]\n",
       "Year                               int64\n",
       "Month                             object\n",
       "ID                                object\n",
       "Community Center Point            object\n",
       "Longitude                        float64\n",
       "Latitude                         float64\n",
       "dtype: object"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_clean['Date'] = pd.to_datetime(df_clean['Date'], format='%Y/%m')\n",
    "df_clean.dtypes"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Now I am going to display the new DataFrame, with no NaN values, Geo-Spacial columns and date-time format column"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Sector</th>\n",
       "      <th>Community Name</th>\n",
       "      <th>Category</th>\n",
       "      <th>Crime Count</th>\n",
       "      <th>Resident Count</th>\n",
       "      <th>Date</th>\n",
       "      <th>Year</th>\n",
       "      <th>Month</th>\n",
       "      <th>ID</th>\n",
       "      <th>Community Center Point</th>\n",
       "      <th>Longitude</th>\n",
       "      <th>Latitude</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>NORTHWEST</td>\n",
       "      <td>ARBOUR LAKE</td>\n",
       "      <td>Theft OF Vehicle</td>\n",
       "      <td>2</td>\n",
       "      <td>10619.0</td>\n",
       "      <td>2022-04-01</td>\n",
       "      <td>2022</td>\n",
       "      <td>APR</td>\n",
       "      <td>2022-APR-ARBOUR LAKE-Theft OF Vehicle</td>\n",
       "      <td>POINT (-114.20767498075155 51.1325947114686)</td>\n",
       "      <td>-114.207675</td>\n",
       "      <td>51.132595</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>CENTRE</td>\n",
       "      <td>BANFF TRAIL</td>\n",
       "      <td>Theft OF Vehicle</td>\n",
       "      <td>2</td>\n",
       "      <td>4153.0</td>\n",
       "      <td>2023-10-01</td>\n",
       "      <td>2023</td>\n",
       "      <td>OCT</td>\n",
       "      <td>2023-OCT-BANFF TRAIL-Theft OF Vehicle</td>\n",
       "      <td>POINT (-114.11512839716917 51.07421633024228)</td>\n",
       "      <td>-114.115128</td>\n",
       "      <td>51.074216</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>EAST</td>\n",
       "      <td>DOVER</td>\n",
       "      <td>Theft OF Vehicle</td>\n",
       "      <td>5</td>\n",
       "      <td>10351.0</td>\n",
       "      <td>2022-12-01</td>\n",
       "      <td>2022</td>\n",
       "      <td>DEC</td>\n",
       "      <td>2022-DEC-DOVER-Theft OF Vehicle</td>\n",
       "      <td>POINT (-113.99305400906283 51.02256772250409)</td>\n",
       "      <td>-113.993054</td>\n",
       "      <td>51.022568</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>CENTRE</td>\n",
       "      <td>GREENVIEW</td>\n",
       "      <td>Assault (Non-domestic)</td>\n",
       "      <td>2</td>\n",
       "      <td>1906.0</td>\n",
       "      <td>2020-12-01</td>\n",
       "      <td>2020</td>\n",
       "      <td>DEC</td>\n",
       "      <td>2020-DEC-GREENVIEW-Assault (Non-domestic)</td>\n",
       "      <td>POINT (-114.05746990262463 51.09485613506574)</td>\n",
       "      <td>-114.057470</td>\n",
       "      <td>51.094856</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>NORTHWEST</td>\n",
       "      <td>HAMPTONS</td>\n",
       "      <td>Theft FROM Vehicle</td>\n",
       "      <td>3</td>\n",
       "      <td>7382.0</td>\n",
       "      <td>2019-08-01</td>\n",
       "      <td>2019</td>\n",
       "      <td>AUG</td>\n",
       "      <td>2019-AUG-HAMPTONS-Theft FROM Vehicle</td>\n",
       "      <td>POINT (-114.14668419231347 51.14509283969437)</td>\n",
       "      <td>-114.146684</td>\n",
       "      <td>51.145093</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>67257</th>\n",
       "      <td>EAST</td>\n",
       "      <td>SOUTHVIEW</td>\n",
       "      <td>Break &amp; Enter - Commercial</td>\n",
       "      <td>1</td>\n",
       "      <td>1805.0</td>\n",
       "      <td>2023-12-01</td>\n",
       "      <td>2023</td>\n",
       "      <td>DEC</td>\n",
       "      <td>2023-DEC-SOUTHVIEW-Break &amp; Enter - Commercial</td>\n",
       "      <td>POINT (-113.99733916298928 51.03415221260387)</td>\n",
       "      <td>-113.997339</td>\n",
       "      <td>51.034152</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>67258</th>\n",
       "      <td>NORTH</td>\n",
       "      <td>SAGE HILL</td>\n",
       "      <td>Theft FROM Vehicle</td>\n",
       "      <td>1</td>\n",
       "      <td>7924.0</td>\n",
       "      <td>2023-12-01</td>\n",
       "      <td>2023</td>\n",
       "      <td>DEC</td>\n",
       "      <td>2023-DEC-SAGE HILL-Theft FROM Vehicle</td>\n",
       "      <td>POINT (-114.14068609335015 51.175616972779984)</td>\n",
       "      <td>-114.140686</td>\n",
       "      <td>51.175617</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>67259</th>\n",
       "      <td>CENTRE</td>\n",
       "      <td>WINSTON HEIGHTS/MOUNTVIEW</td>\n",
       "      <td>Break &amp; Enter - Commercial</td>\n",
       "      <td>1</td>\n",
       "      <td>3635.0</td>\n",
       "      <td>2023-12-01</td>\n",
       "      <td>2023</td>\n",
       "      <td>DEC</td>\n",
       "      <td>2023-DEC-WINSTON HEIGHTS/MOUNTVIEW-Break &amp; Ent...</td>\n",
       "      <td>POINT (-114.04184874950579 51.075298802175126)</td>\n",
       "      <td>-114.041849</td>\n",
       "      <td>51.075299</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>67260</th>\n",
       "      <td>CENTRE</td>\n",
       "      <td>SOUTH CALGARY</td>\n",
       "      <td>Street Robbery</td>\n",
       "      <td>1</td>\n",
       "      <td>4442.0</td>\n",
       "      <td>2023-12-01</td>\n",
       "      <td>2023</td>\n",
       "      <td>DEC</td>\n",
       "      <td>2023-DEC-SOUTH CALGARY-Street Robbery</td>\n",
       "      <td>POINT (-114.10207470148995 51.02680215136445)</td>\n",
       "      <td>-114.102075</td>\n",
       "      <td>51.026802</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>67261</th>\n",
       "      <td>NORTHEAST</td>\n",
       "      <td>SKYLINE EAST</td>\n",
       "      <td>Break &amp; Enter - Commercial</td>\n",
       "      <td>1</td>\n",
       "      <td>0.0</td>\n",
       "      <td>2023-12-01</td>\n",
       "      <td>2023</td>\n",
       "      <td>DEC</td>\n",
       "      <td>2023-DEC-SKYLINE EAST-Break &amp; Enter - Commercial</td>\n",
       "      <td>POINT (-114.03893128621468 51.10015261262966)</td>\n",
       "      <td>-114.038931</td>\n",
       "      <td>51.100153</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>67193 rows × 12 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "          Sector             Community Name                    Category  \\\n",
       "0      NORTHWEST                ARBOUR LAKE            Theft OF Vehicle   \n",
       "1         CENTRE                BANFF TRAIL            Theft OF Vehicle   \n",
       "2           EAST                      DOVER            Theft OF Vehicle   \n",
       "3         CENTRE                  GREENVIEW      Assault (Non-domestic)   \n",
       "4      NORTHWEST                   HAMPTONS          Theft FROM Vehicle   \n",
       "...          ...                        ...                         ...   \n",
       "67257       EAST                  SOUTHVIEW  Break & Enter - Commercial   \n",
       "67258      NORTH                  SAGE HILL          Theft FROM Vehicle   \n",
       "67259     CENTRE  WINSTON HEIGHTS/MOUNTVIEW  Break & Enter - Commercial   \n",
       "67260     CENTRE              SOUTH CALGARY              Street Robbery   \n",
       "67261  NORTHEAST               SKYLINE EAST  Break & Enter - Commercial   \n",
       "\n",
       "       Crime Count  Resident Count       Date  Year Month  \\\n",
       "0                2         10619.0 2022-04-01  2022   APR   \n",
       "1                2          4153.0 2023-10-01  2023   OCT   \n",
       "2                5         10351.0 2022-12-01  2022   DEC   \n",
       "3                2          1906.0 2020-12-01  2020   DEC   \n",
       "4                3          7382.0 2019-08-01  2019   AUG   \n",
       "...            ...             ...        ...   ...   ...   \n",
       "67257            1          1805.0 2023-12-01  2023   DEC   \n",
       "67258            1          7924.0 2023-12-01  2023   DEC   \n",
       "67259            1          3635.0 2023-12-01  2023   DEC   \n",
       "67260            1          4442.0 2023-12-01  2023   DEC   \n",
       "67261            1             0.0 2023-12-01  2023   DEC   \n",
       "\n",
       "                                                      ID  \\\n",
       "0                  2022-APR-ARBOUR LAKE-Theft OF Vehicle   \n",
       "1                  2023-OCT-BANFF TRAIL-Theft OF Vehicle   \n",
       "2                        2022-DEC-DOVER-Theft OF Vehicle   \n",
       "3              2020-DEC-GREENVIEW-Assault (Non-domestic)   \n",
       "4                   2019-AUG-HAMPTONS-Theft FROM Vehicle   \n",
       "...                                                  ...   \n",
       "67257      2023-DEC-SOUTHVIEW-Break & Enter - Commercial   \n",
       "67258              2023-DEC-SAGE HILL-Theft FROM Vehicle   \n",
       "67259  2023-DEC-WINSTON HEIGHTS/MOUNTVIEW-Break & Ent...   \n",
       "67260              2023-DEC-SOUTH CALGARY-Street Robbery   \n",
       "67261   2023-DEC-SKYLINE EAST-Break & Enter - Commercial   \n",
       "\n",
       "                               Community Center Point   Longitude   Latitude  \n",
       "0        POINT (-114.20767498075155 51.1325947114686) -114.207675  51.132595  \n",
       "1       POINT (-114.11512839716917 51.07421633024228) -114.115128  51.074216  \n",
       "2       POINT (-113.99305400906283 51.02256772250409) -113.993054  51.022568  \n",
       "3       POINT (-114.05746990262463 51.09485613506574) -114.057470  51.094856  \n",
       "4       POINT (-114.14668419231347 51.14509283969437) -114.146684  51.145093  \n",
       "...                                               ...         ...        ...  \n",
       "67257   POINT (-113.99733916298928 51.03415221260387) -113.997339  51.034152  \n",
       "67258  POINT (-114.14068609335015 51.175616972779984) -114.140686  51.175617  \n",
       "67259  POINT (-114.04184874950579 51.075298802175126) -114.041849  51.075299  \n",
       "67260   POINT (-114.10207470148995 51.02680215136445) -114.102075  51.026802  \n",
       "67261   POINT (-114.03893128621468 51.10015261262966) -114.038931  51.100153  \n",
       "\n",
       "[67193 rows x 12 columns]"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_clean"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "# df_clean.to_csv(\"Data.csv\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Now I am going to calculate 'Crime Rate per 1000 people'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "crime_per_community = df_clean.groupby('Community Name').agg(\n",
    "    total_crimes=pd.NamedAgg(column='Crime Count', aggfunc='sum'),\n",
    "    resident_count=pd.NamedAgg(column='Resident Count', aggfunc='mean')\n",
    ")\n",
    "\n",
    "crime_per_community['crime_rate_per_1000'] = (crime_per_community['total_crimes'] / crime_per_community['resident_count']) * 1000\n",
    "crime_per_community_sorted = crime_per_community.sort_values(by='crime_rate_per_1000', ascending=False)\n",
    "crime_per_community = crime_per_community[crime_per_community['resident_count'] > 0]\n",
    "crime_per_community['resident_count'] = crime_per_community['resident_count'].replace(0, crime_per_community['resident_count'].mean())\n",
    "crime_per_community['resident_count'] = crime_per_community['resident_count'].fillna(crime_per_community['resident_count'].mean())\n",
    "crime_per_community['crime_rate_per_1000'] = crime_per_community['crime_rate_per_1000'].round(2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "# crime_per_community.to_csv(\"Data_2.csv\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Questions to explore the data and Visualization"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## In this section I am going to ask 5 questions, which will be a guide to explore the data"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "1. How are the categories of crimes distributed?\n",
    "\n",
    "2. What are the sectors with the highest number of crimes and How are crimes distributed throughout the city?\n",
    "\n",
    "3. How have crimes evolved over time?\n",
    "\n",
    "4. What is the crime rate per capita?\n",
    "\n",
    "5. What is the relationship between Crimes and population?"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Crimes by Category (Question 1)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "To answer this question, I am going to create a pie chart to show the categories of crimes and their percentages "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Category</th>\n",
       "      <th>Crime Count</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Assault (Non-domestic)</td>\n",
       "      <td>24021</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Break &amp; Enter - Commercial</td>\n",
       "      <td>25868</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Break &amp; Enter - Dwelling</td>\n",
       "      <td>11468</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Break &amp; Enter - Other Premises</td>\n",
       "      <td>11987</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Commercial Robbery</td>\n",
       "      <td>2029</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>Street Robbery</td>\n",
       "      <td>3636</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>Theft FROM Vehicle</td>\n",
       "      <td>71187</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>Theft OF Vehicle</td>\n",
       "      <td>31045</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>Violence Other (Non-domestic)</td>\n",
       "      <td>12390</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                         Category  Crime Count\n",
       "0          Assault (Non-domestic)        24021\n",
       "1      Break & Enter - Commercial        25868\n",
       "2        Break & Enter - Dwelling        11468\n",
       "3  Break & Enter - Other Premises        11987\n",
       "4              Commercial Robbery         2029\n",
       "5                  Street Robbery         3636\n",
       "6              Theft FROM Vehicle        71187\n",
       "7                Theft OF Vehicle        31045\n",
       "8   Violence Other (Non-domestic)        12390"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_clean.groupby('Category')['Crime Count'].sum().reset_index()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAABPwAAAPOCAYAAACbKAbeAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8pXeV/AAAACXBIWXMAAA9hAAAPYQGoP6dpAAEAAElEQVR4nOzdd1hT598G8DuEEQhbURBRVETEveuoYqs/Z9XWuqq1WHerVq2jrVq1an211WpttW7cFrXauvfeCxeIgCCgIHsTIMl5/7CkRoagwEnC/bmuXA3nPOec+wRLwpdnSARBEEBEREREREREREQGwUjsAERERERERERERFRyWPAjIiIiIiIiIiIyICz4ERERERERERERGRAW/IiIiIiIiIiIiAwIC35EREREREREREQGhAU/IiIiIiIiIiIiA8KCHxERERERERERkQFhwY+IiIiIiIiIiMiAsOBHRERERERERERkQFjwIyIiItGFhYVBIpFAIpHAx8dH7Dhv5MyZM5p7OHPmjNhxiIiIiKgcY8GPiIiI3trLxa5XH+bm5nBxcUHPnj2xbt06KBQKseMSERERERk0FvyIiIioVCkUCkRGRuLgwYMYOXIkGjdujEePHokdi16SW5ydM2eO2FGIiIiIqAQYix2AiIiIDMvYsWPxxRdfaL7OyMiAn58fli1bhoCAAAQGBqJr16548OABzM3NAQCurq4QBEGsyEREREREBoU9/IiIiKhEVapUCfXr19c8WrZsiVGjRuHmzZto2bIlACA0NBTr168XOSkRERERkWFiwY+IiIjKhLm5ORYsWKD5+vDhwyKmISIiIiIyXCz4ERERUZl55513NM+fPHmieV6cVXqvXbuGkSNHwt3dHZaWlpDL5fDw8MCXX36JoKCgEst68eJFjBgxAnXq1IG1tTUsLS3h4eGBPn36YPPmzUhJSXntOXx9ffH+++/DwcEB5ubmqFOnDqZNm4aEhIRCj7ty5QpmzpwJLy8vODo6wtTUFNbW1vD09MTYsWPh7+9f6PHe3t6QSCRwdXUFAERFRWH69OmoV68erKysNCsJu7q6QiKRaI6bO3dunkVXvL29X3ufRERERKRbOIcfERERlRlj4/8+eqhUqmIdq1QqMWHCBKxatSrPvsDAQAQGBmLt2rX4/fffMXLkyDfOmJmZieHDh2PHjh0FXufvv//G7Nmz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oW/fvpqvP/vsM6F3794FticiInodTiRBRERERDqtY8eOWLVqFQAgISEBK1euRLdu3XDt2jVUr15d06558+aa57GxsYiIiMDw4cMxcuRIzXalUgkbGxvN17t378ayZcsQHByMtLQ0KJVKzfDZXOHh4ejUqRPmz5+PSZMmFZrV398fPj4+ePDgAerWrYthw4bBy8sLR44cQaVKlfDgwQN8+umnhZ5j8ODBmDFjhta2V3vaNWzYUPPcyckJABATEwMPD498z3nr1i0IggB3d3et7VlZWVo9JU1NTbXOnZ969epphvq+++67WkNpcwmCUOg5cgUEBKB169aQSCSabW3btkVaWhoiIyNRrVo1ANr3K5VKUaFCBTRo0ECzLbc3XUxMjNb5Xz6ucuXKmqHbL2+7du0agBffO4VCgc6dO2udIzs7G02aNNHa9vK/tYCAAGRlZeH9998v0j0DwB9//IF169bhyZMnyMzMRHZ2NleDJiKiEsWCHxERERHpNLlcDjc3N83XzZo1g42NDdauXas19FMul2ue586vt3btWrRq1UrrfFKpFABw5coVDBw4EHPnzkWXLl1gY2ODnTt3YsmSJVrtHRwcUKVKFezcuRPDhw/PUxB82d27d2FqagpPT08AwPr16zFgwAC0bdsWU6dORWpqKnr16lXo/drY2Gjdb35MTEw0z3OLZYUtsKFWqyGVSnHz5k3N/eeytLTUPDc3N9cqvuXn0KFDmnkOzc3N822TW1h8+PAhWrduXeC5BEHIc73cYuHL21++39x9RXkNXm2T33lyj8n978GDB+Hs7KzVzszMTOvrl/+tFfQaFMTX1xeTJk3CkiVL0Lp1a1hZWeGnn37C1atXi3UeIiKiwrDgR0RERER6RSKRwMjICJmZmQW2qVy5MpydnfH48WMMHjw43zYXL15E9erVtXrT5bdIhbm5OQ4cOIDu3bujS5cuOHbsGKysrPI9p7OzM7Kzs3H16lW0atUKUqkU27dvR+/evTF69GgsXbq02AWi4jI1NYVKpdLa1qRJE6hUKsTExODdd999q/O/3KuyIP/73/9QsWJFLF68GHv37s2zPykpCba2tvD09MSePXu0Cn+XLl2ClZVVnqJbactdqCQ8PBwdOnQo8nG1a9eGubk5Tp48iREjRry2/fnz59GmTRt88cUXmm0hISFvlJmIiKggLPgRERERkU7LyspCdHQ0ACAxMRG//fYb0tLS8MEHHxR63Jw5czBhwgRYW1ujW7duyMrKwo0bN5CYmIjJkyfDzc0N4eHh2LlzJ1q0aIGDBw/mW5wCXvToOnjwILp164Zu3brhyJEjWj3jcrVr1w5t2rTBgAEDsGzZMjRo0AD37t3D48ePIZfLsX37dowePRoWFhYF5s7IyNDcby4zMzPY2dm97qUCALi6uiItLQ0nT55Eo0aNYGFhAXd3dwwePBhDhw7FkiVL0KRJE8TFxeHUqVNo0KABunfvXqRzF5VcLse6devQr18/9OrVCxMmTICbmxvi4uLg6+ured2/+OILLFu2DOPHj8e4ceMQGBiI2bNnY/LkyQUuilJarKysMGXKFEyaNAlqtRrt2rVDSkoKLl26BEtLS3z22Wf5HieTyTB9+nRMmzYNpqamaNu2LWJjY/HgwQMMHz48T3s3Nzds3rwZR48eRY0aNbBlyxZcv34dNWrUKO1bJCKicoSr9BIRERGRTjty5AicnJzg5OSEVq1a4fr169i1axe8vLwKPW7EiBFYt24dfHx80KBBA3To0AE+Pj6awkrv3r0xadIkjBs3Do0bN8alS5cwa9asAs9naWmJw4cPQxAEdO/eHenp6XnaSCQSHDlyBH379sXkyZPh6emJGTNmYOzYsXj06BGio6MxePDgQoffrl27VnO/uY9BgwYV7cUC0KZNG4wZMwYDBgyAg4MDFi9eDADYuHEjhg4diq+//hp16tRBr169cPXqVbi4uBT53MXRu3dvXLp0CSYmJvjkk0/g4eGBQYMGITk5WTMU29nZGYcOHcK1a9fQqFEjjBkzBsOHD8fMmTNLJdPrzJs3D99//z0WLlyIunXrokuXLti/f/9ri3GzZs3C119/je+//x5169bFgAED8swnmGvMmDH46KOPMGDAALRq1Qrx8fFavf2IiIhKgkQo6oy6REREREREREREpPPYw4+IiIiIiIiIiMiAsOBHRERERERERERkQFjwIyIiIiIiIiIiMiAs+BERERERERERERkQFvyIiIiIiIiIiIgMCAt+REREREREREREBoQFPyIiIiIiIiIiIgPCgh8REREREREREZEBYcGPiIiIiIiIiIjIgLDgR0REREREREREZEBY8CMiIiIiIiIiIjIgLPgREREREREREREZEBb8iIiIiIiIiIiIDAgL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xWljUKhPxuEOCR0e/3mvQUREA0+vUU7BDzi5ecdDDz2EP//5z3jzzTcxZMgQXHbZZed9zlcLYbIsn7M4FggEMG/ePLz00ktnPHa6QAYAWq32jHsETm2uYzR2XafzbPf43ve+h0ceeeSMxwYPHnzW52RnZ6OgoOCsjxUVndz4Kysr67z3/Wru0/8NTufubi6z2XzB+yQkJKClpeW85/zmN7/B9OnT8dRTT3U5fq4B0q/+vZ3t7+D0cydPntzZ5QmcLL4dPXr0grn7cu/z/ZsIBAJQq9XIz88/Y7OM00XC73znO7jyyiuxbNkyrF69Gi+++CJeeeUVPPzwwxfMDVz4391pzc3NXYqVF6Ks7xAU2djhR0REIUil1sI4ZtKFT1QIndOGnL0rcf2yl/HMwifx1pqn8Yea/+EhbTnmxvmRatFe+CJERBRUapUETQitIdsdt956K9RqNd5//328/fbbuOeee85ZvLNarUhLS8PmzZu7HN+6des5u+gmTpyIw4cPIzMzEyNGjOjypzuFLuBkF111dTWOHDly3nt89fojRow46463AHDbbbdhzZo1Z3QmBgIBvPbaaxg5cmTn+n46na7LZhHd1Ztc5zJhwoRzFihPmzp1Km688UY8++yzXY6PHDkSPp8PO3bs6DzW1NSEI0eOdLv70Wg0dskfFRWFESNGQKvVYvv27Z3ntbS0dPl76o97n82ECRPg9/vR0NBwxn/blJQvdm7PyMjAAw88gIULF+KJJ57AP//5TwDo/O9/vr/XMWPGIBAIXHBDjkOHDmHChAndzq6s7xAU2RT2A42IiCKHKXvchU9SKFXAj0ElO/G1z/6CHyz8Ef786eP4d+Hf8KPAAdwY40BOjBYKWTeeiChsGBTW3Qec7Ib6xje+gR//+Meora3t3K30XJ566im89NJLWLBgAYqLi/Hss89i3759ePTRR896/oMPPojm5mbcfvvt2LlzJ44dO4bVq1fj3nvv7XYRbc6cObj44otx00034bPPPkNZWRlWrFjRuTvwM888g23btuHBBx/Evn37UFJSgqVLl563k+vxxx/H1KlTMW/ePHz00UeorKzErl27cNNNN6GwsBD/+te/OgufmZmZOHDgAIqLi9HY2Nhls4nz6U2uc7nyyitx+PDhC3b5/frXv8a6detQXFzceSwrKws33HAD7r//fmzevBn79+/H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WYd9n2VlZbr//vt7nQ1A36LwA5AaMjOl5lGS3286CQAgyfkCYfkG1ZiOASRM5PTzlf2FW11X9lmWddDT5ZdfHrfHevTRR/Xiiy/qlVde0cqVK7Vhw4ZuFYmPPvqoRo8erXA4rIyMDI0fP15PPfVUp+s8//zzXT6Pr371q/vd365du5SXl6dvfvObXT7ePffco7y8PO3atavbz/XTzjvvPM2fP7/X9wMguVH4AUgd0ag0drQUCplOAgBIcuH6kaYjAAmRMelSZV9zs+kYCbFy5cq9p/vvv1/RaLTTsQceeCBuj7Vo0SLV1dWpoaFBRUVF3SpPb775Zl1zzTU699xz9c477+iNN97QUUcdpTPOOEMPPfTQftefN29ep+dx22237Xcdv9+viy++WNOnT1dHR8d+lz/yyCO65JJL5I/Dl9+hUEgFBQW9vh8AyY3CD0BqCYdjpV8kYjoJACCJOcWDJNY1g8tkTr5eWZOvNx0jYYqKivaeMjMzZVnWfsc+tnjxYh1zzDFyHEdNTU169dVXO93XK6+8ovHjxysUCqm0tFTXX3+9tm3bJkk6+uijdd999+lf//qXLMvS0UcfrfLycknSsGHD9h7rymuvvab77rtP3/3ud3XzzTersrJSdXV1mjZtmm688UZ96Utf0vLlyzvdpqCgoNPziBzgfeyUKVO0aNEi/etf/+p0/MUXX9SCBQs0ZcoUSbHyr66uTsFgULW1tXr44Yf3u6+DvT5dTen905/+pJEjRyoYDCovL09nn312lxkladOmTbr66qtVUFCgaDSqY489Vu+8884Brw/ADN4FAUg9waA0ZlRsbT8AALrg8QUVqGkwHQOID9uj7BvuUHTSpaaTJI3bb79dN998s95++21VV1frggsuUGtrqyTpvffe04knnqizzz5b7777rh577DG99NJLuu666yRJTzzxhK666iqNHTtWK1eu1BNPPKE33nhDkvTss8/uPdaV3/72t4pEIrrmmmv2u2zq1KnavXu3/vCHP/ToOTU2NmrUqFF65JFHOh3/xS9+odGjR6uhoUE//elPdfvtt2vatGmaM2eOvvWtb+mOO+7Qo48+etivz6f95S9/0dlnn62JEyfqrbfe0j/+8Q+NHNn1KOmOjg5NnDhRq1at0tNPP60ZM2Zo+PDhOu6447R+/foePW8AicEK+ABSk98f2733P29L6z4ynQYAkIScuhFqmfOu6RhA7/j8yr31W3LGHm06SVK5+eabNXHiREnS3Xffrfr6ei1cuFC1tbX67ne/qwsvvFA33nijJKmqqkoPPvigJkyYoB/96EfKycmR4zjy+/0qKiqSJG3evFmSlJubu/dYV+bPn6+Kiooup9aWlJQoMzNzv/Xx+vfv3+n8+++/r9zc3C7vf/Lkybr55pv10EMPKRKJaOvWrXr88cf1/e9/X5L0jW98Q/fdd9/eEXjl5eWaPXu2fvKTn+iyyy47rNfn06ZNm6bzzz9fd999995jTU1NXeZ77rnn9N5772nNmjUKBAKSpO9973t68skn9b//+7+6+uqru7wdgL7HCD8AqcvrlUYOl4oKTScBACShUMHA2N8KIEVZobDy736Qsq8LQ4YM2ft7cXGxJGnNmjWSpBkzZmj69OmKRCJ7TyeeeKLa29u1ZMmShObq6OjYbz3AF198UW+//fbeU3Z29gFvf8EFF6i9vV2PPfaYJOmxxx5TR0eHzj//fK1du1bLly/XlClTOj23b37zm1q0aFGn+znY6/Npb7/9to477rjDen4zZszQ1q1blZub2ynDkiVL9ssAwCzeAaW5pUuXqry8XG+99ZaGDh3a4/vZvn27LrnkEv3973/Xli1btGHDhm5t9Z5upk+frhtvvFEbN2484HUuv/xybdy4UU8++eQh7y9e/zumJNuWhjVJM2dLyz8wnQYAkEQ8Hp+C9cO18503TEcBus3OzFb+3Q/KX1VnOkpS8vl8e3//uGBrb2/f+/Oaa67R9dfvv97hgAEDevW41dXVeumll7Rr1679RvmtWLFCmzdvVlVVVafj5eXlh/3ZKDMzU+ecc44eeeQRTZkyRY888ojOOeccRaNRrV69WpL005/+VM3NzZ1u5/F4Op0/2OvzaaFubIjX3t6u4uJiPf/88/tdxuc/ILkwws/FUmFb+6VLl3aZ7eKLL+7y8szMTI0ZM0Z//vOf97uvHTt26K677lJNTY0CgYDy8vJ0zjnnaNasWZ2u97WvfU2WZemkk07a7z6+853vHHSR3hkzZsiyLL300ktdXn7iiSfq9NNPP+hzPlwPPPCApk+fHpf7cj3LkhrrpUHlppMAAJKMUz3UdASg2zz5RSr47s8o+3po+PDhmjVrliorK/c7HWiX24+Pt7W1HfS+zz//fG3dulU/+clP9rvse9/7nnw+nyZNmtSr/FOmTNHLL7+sp556Si+//PLezToKCwvVr18/LV68eL/n9fGmIz0xZMgQ/eMf/zis6w4fPlyrVq2S1+vdL0NeXl6PMwCIP0b4udjKlSv3/v7YY4/pzjvv1Lx58/YeC4VC2rBhQ1wea99t7aVYUdcdzz77rOrr6ztl6+ryjRs36uGHH9akSZP0n//8Z+/jtbS06Pjjj9eyZct03333qbm5WatXr9Y999yj5uZmPfvssxozZsze+ysuLtZzzz2nDz74oNOaGo888shBv/UbMWKEmpqa9Mgjj2jcuHGdLlu+fLmeffbZAy7w21377kKGw1RbLfl90tz5h74uACAthPL6ywqG1LFzh+kowGHxlpYr/5sPyZvHkiU9deutt2rMmDG69tprddVVVykcDmvOnDn6+9//rh/+8Idd3qagoEChUEjPPPOM+vfvr2Aw2OX78bFjx+qGG27Ql7/8Ze3atUtnnnmmdu/erV/96ld64IEHdP/996u0tLRX+SdMmKDKykpdeumlqqys1Pjx4/de9rWvfU3XX3+9otGoTj75ZLW0tOjf//63NmzYoC996Us9ery77rpLxx13nCoqKnT++eertbVV//d//6dbbrllv+sef/zxGjt2rM4880zde++9qqmp0YoVK/T000/rzDPPPOBmHwD6HiP8XCwVtrX/2MeL43aVbd/La2trNW3aNO3evVvPPffc3svvv/9+vfrqq3rqqad07rnnauDAgRo9erT+8Ic/qK6uTlOmTFFHR8fe6xcUFOiEE07otJvVK6+8onXr1u1d3PZApkyZot///vd7n//Hpk+frvz8fE2cOFG7du3SLbfcon79+ikcDqu5ubnLYe9//etfVVdXp0gkopNOOqlTSXv55ZfrzDPP3Hu+vb1d9957ryorKxUIBDRgwABNmzbtgDlnz56tU045RZFIRIWFhbrkkku0bt26gz43VxhULjU2xEb9AQDSnm17FBoyynQM4LD4q+tV8J2fUvb10pAhQ/TCCy9owYIFOuqoozRs2DDdcccde9ey64rX69WDDz6on/zkJyopKdEZZ5xxwOvef//9evjhh/W73/1OjY2NGjFihF544QU9+eST+uIXvxiX5zB58mRt2LBBkydP7nT8yiuv1M9+9jNNnz5djY2NmjBhgqZPn96rEX5HH320Hn/8cf3pT3/S0KFDdeyxx+r111/v8rqWZenpp5/W+PHjNXnyZFVXV+v888/X0qVLVVjI/2+BZGJ17NuCwLUOtGbcx2u/1dbW6nvf+56qqqp0++23680339TChQvl9Xr13nvv6YgjjtA3vvENTZw4UWvXrtV11123d6Tb+vXrddttt2nmzJl64okn5Pf7tWjRIo0ePXrvyDy/36+cnJz9ch1q7blPX75792798Ic/1NSpU/WjH/1In/vc5yTFdpEqKirSX//61/3u4ze/+Y0uuuiivffxta99TU8++aTuuusu3XLLLVqwYIGk2B/PSCQiKbZwbVcFnSStX79eJSUl+vGPf7x3WnRHR4cqKyt1zjnn6N5779VFF12kpUuX6tvf/rZKSkr0xz/+UV/96lf13nvvqaqqStOnT9fVV1+tCRMm6J577pFt27r44os1bNgw/frXv5a0/xp+t956q37605/qBz/4gcaNG6eVK1dq7ty5uvLKK/d7nVauXKkhQ4boqquu0qWXXqodO3bo1ltvVWtrq/75z38e6v8u7rB6jfT2u9IhpmUAANxvx4YVWveb/zIdAzio4MgjlHvbt2WHHNNRAAAuwAg/SPpk2/bq6mrdfffdev/997Vw4UJJ6rStfVVVlY444gg9+OCD+uUvf6mdO3fut619Tk6O8vPzJX0yMq+rsm9fRxxxRKddnt56660uLw8Gg5o6darKysp07rnn7r18/vz5qqvreo2Tj4/Pn995muepp56qzZs361//+pe2bdum3//+9/t9g9aVnJwcnXnmmXrkkUf2Hnv++ee1ePFiTZ48WYsWLdJvf/tbPf744zrqqKNUUVGhm2++WePGjet0m927d+vHP/6xRo4cqeHDh+u666474NoZW7Zs0QMPPKDvfOc7uuyyy1RRUaFx48bpyiuv7PL6P/rRjzR8+HB961vfUm1trYYNG6Zf/OIXeu655/Z7HVyrsEBqHiUdYJ0WAED6CGYWyY5ETccADih8yjnKu/MHlH0AgLhhDT9IOvC27bW1tZoxY4YWLly4d+SZFBvR9vG29gcq2rrjscce63Q/n1734rHHHlNtba3mz5+vG2+8UT/+8Y8PWSLum1X6ZHeqj/l8Pl188cV65JFHtHjxYlVXV3d6HQ5mypQpOuGEE7Rw4UJVVlbqF7/4hY488kjV1NTo8ccfV0dHh6qrqzvdpqWlRbm5uXvPO46jioqKveeLi4u1Zs2aLh9vzpw5amlp0XHHHXdY+WbMmKHnnntu74jFfS1atGi/bK6VlSkdOUZ68z/S1q2m0wAADLFsW6Ghzdr20t9NRwE6syxlXnG9opMuMZ0EAOAyFH6QZG5b+4+VlpaqsrLyoJdXVVWpqqpKkUhEkyZN0uzZs1VQUCBJqq6u1uzZs7u87dy5cyVJVVVV+102efJkNTc3a+bMmYc1uu9jxx9/vAYOHKjp06frlltu0RNPPKGHHnpIUuz18ng8mjFjhjweT6fb7VvA7fuaS7HX/UAz7D+9icmhtLe367TTTtO9996732UHW7vElUIhaexo6T/vSB99ZDoNAMAQp7yewg9JxQoElDP163KOPLwvdAEA6A4KPxzSvtvaH67D3da+JyZMmKCGhgZNmzZNDzzwgCTp/PPP1+2336533nlHTU1Ne6/b3t6uH/zgBxo8eHCn4x+rr69XfX293n33XV144YWHncGyLF1xxRX62c9+pv79+8u27b1TjIcNG6a2tjatWbNGRx11VC+fbUxVVZVCoZD+8Y9/HHAa776GDx+uP/zhDyorK5PXy3/m8vmkUcOlmbOlDz40nQYAYEAgI1eenDy1rU+DDayQ9OzMbOXd+X0FahtNRwEAuBRr+OGQbr31Vr366qu69tpr9fbbb2vBggX605/+dNAdqPbd1n716tXatGlTXDNNnTpVP/nJT/Thh7Hy5qabbtLo0aN12mmn6fHHH9eyZcv05ptvatKkSZozZ45+/vOf7zel92P//Oc/tXLlSmVlZXUrwxVXXKEVK1boK1/5is4//3yFw2FJsdGGF110kS699FI98cQTWrJkid58803de++9evrpp3v0fIPBoG699Vbdcsst+uUvf6lFixbptdde089//vMur3/ttddq/fr1uuCCC/TGG29o8eLF+tvf/qbJkycnpIRNCbYtDWmQqvcf6QkAcD/LshVqGmM6BiBv/4EqvO8Ryj4AQEJR+OGQEr2tfU+ceuqpKisr07Rp0yTFCrF//vOfuuyyy/SVr3xFlZWVOumkk+TxePTaa69pzJgDv8EPh8PdLvuk2HTm448/Xhs2bNhvOvAjjzyiSy+9VFOnTlVNTY1OP/10vf766/utTdgdd9xxh6ZOnao777xTdXV1Ou+88w645l9JSYlefvlltbW16cQTT1RD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",
      "text/plain": [
       "<Figure size 1500x1200 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "a = df_clean.groupby('Category')['Crime Count'].sum().reset_index()\n",
    "etiquetas = a['Category']\n",
    "valores = a['Crime Count']\n",
    "\n",
    "plt.style.use('ggplot')\n",
    "plt.figure(figsize=(15, 12))  \n",
    "plt.pie(valores, labels=etiquetas, autopct='%1.1f%%', startangle=10)  \n",
    "plt.title('Pie chart', size=20) \n",
    "plt.axis('equal') \n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "As we can see here, 'Theft From Vehicle' represents 36.8% of Calgary´s crimes, followed by 'Theft of Vehicle'. Both of them combine represent half of the crimes in Calgary"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Types of Crime by Sector (Question 2)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "To answer this question, I am going to plot 2 charts. The first one showing crimes by sector and types, and the second one a geo-spacial plot of Calgary"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/vnd.plotly.v1+json": {
       "config": {
        "plotlyServerURL": "https://plot.ly"
       },
       "data": [
        {
         "marker": {
          "line": {
           "color": "rgb(248, 248, 249)",
           "width": 1
          }
         },
         "name": "Assault (Non-domestic)",
         "type": "bar",
         "x": [
          "CENTRE",
          "EAST",
          "NORTH",
          "NORTHEAST",
          "NORTHWEST",
          "SOUTH",
          "SOUTHEAST",
          "WEST"
         ],
         "y": [
          9516,
          2304,
          1159,
          4574,
          2040,
          2343,
          961,
          1124
         ]
        },
        {
         "marker": {
          "line": {
           "color": "rgb(248, 248, 249)",
           "width": 1
          }
         },
         "name": "Break & Enter - Commercial",
         "type": "bar",
         "x": [
          "CENTRE",
          "EAST",
          "NORTH",
          "NORTHEAST",
          "NORTHWEST",
          "SOUTH",
          "SOUTHEAST",
          "WEST"
         ],
         "y": [
          11482,
          2061,
          1240,
          3628,
          1806,
          2658,
          1769,
          1224
         ]
        },
        {
         "marker": {
          "line": {
           "color": "rgb(248, 248, 249)",
           "width": 1
          }
         },
         "name": "Break & Enter - Dwelling",
         "type": "bar",
         "x": [
          "CENTRE",
          "EAST",
          "NORTH",
          "NORTHEAST",
          "NORTHWEST",
          "SOUTH",
          "SOUTHEAST",
          "WEST"
         ],
         "y": [
          2465,
          691,
          1253,
          1510,
          1617,
          1679,
          943,
          1310
         ]
        },
        {
         "marker": {
          "line": {
           "color": "rgb(248, 248, 249)",
           "width": 1
          }
         },
         "name": "Break & Enter - Other Premises",
         "type": "bar",
         "x": [
          "CENTRE",
          "EAST",
          "NORTH",
          "NORTHEAST",
          "NORTHWEST",
          "SOUTH",
          "SOUTHEAST",
          "WEST"
         ],
         "y": [
          4767,
          956,
          671,
          1817,
          1122,
          1519,
          437,
          698
         ]
        },
        {
         "marker": {
          "line": {
           "color": "rgb(248, 248, 249)",
           "width": 1
          }
         },
         "name": "Commercial Robbery",
         "type": "bar",
         "x": [
          "CENTRE",
          "EAST",
          "NORTH",
          "NORTHEAST",
          "NORTHWEST",
          "SOUTH",
          "SOUTHEAST",
          "WEST"
         ],
         "y": [
          564,
          190,
          145,
          468,
          187,
          240,
          88,
          147
         ]
        },
        {
         "marker": {
          "line": {
           "color": "rgb(248, 248, 249)",
           "width": 1
          }
         },
         "name": "Street Robbery",
         "type": "bar",
         "x": [
          "CENTRE",
          "EAST",
          "NORTH",
          "NORTHEAST",
          "NORTHWEST",
          "SOUTH",
          "SOUTHEAST",
          "WEST"
         ],
         "y": [
          1216,
          398,
          207,
          981,
          251,
          294,
          102,
          187
         ]
        },
        {
         "marker": {
          "line": {
           "color": "rgb(248, 248, 249)",
           "width": 1
          }
         },
         "name": "Theft FROM Vehicle",
         "type": "bar",
         "x": [
          "CENTRE",
          "EAST",
          "NORTH",
          "NORTHEAST",
          "NORTHWEST",
          "SOUTH",
          "SOUTHEAST",
          "WEST"
         ],
         "y": [
          23260,
          4991,
          4895,
          13189,
          6587,
          9733,
          4307,
          4225
         ]
        },
        {
         "marker": {
          "line": {
           "color": "rgb(248, 248, 249)",
           "width": 1
          }
         },
         "name": "Theft OF Vehicle",
         "type": "bar",
         "x": [
          "CENTRE",
          "EAST",
          "NORTH",
          "NORTHEAST",
          "NORTHWEST",
          "SOUTH",
          "SOUTHEAST",
          "WEST"
         ],
         "y": [
          7340,
          3579,
          2416,
          8556,
          2265,
          3383,
          2116,
          1390
         ]
        },
        {
         "marker": {
          "line": {
           "color": "rgb(248, 248, 249)",
           "width": 1
          }
         },
         "name": "Violence Other (Non-domestic)",
         "type": "bar",
         "x": [
          "CENTRE",
          "EAST",
          "NORTH",
          "NORTHEAST",
          "NORTHWEST",
          "SOUTH",
          "SOUTHEAST",
          "WEST"
         ],
         "y": [
          3642,
          932,
          1065,
          2153,
          1268,
          1589,
          875,
          866
         ]
        }
       ],
       "layout": {
        "barmode": "stack",
        "height": 1000,
        "legend": {
         "title": {
          "text": "Category"
         }
        },
        "plot_bgcolor": "rgba(0,0,0,0)",
        "template": {
         "data": {
          "bar": [
           {
            "error_x": {
             "color": "#2a3f5f"
            },
            "error_y": {
             "color": "#2a3f5f"
            },
            "marker": {
             "line": {
              "color": "#E5ECF6",
              "width": 0.5
             },
             "pattern": {
              "fillmode": "overlay",
              "size": 10,
              "solidity": 0.2
             }
            },
            "type": "bar"
           }
          ],
          "barpolar": [
           {
            "marker": {
             "line": {
              "color": "#E5ECF6",
              "width": 0.5
             },
             "pattern": {
              "fillmode": "overlay",
              "size": 10,
              "solidity": 0.2
             }
            },
            "type": "barpolar"
           }
          ],
          "carpet": [
           {
            "aaxis": {
             "endlinecolor": "#2a3f5f",
             "gridcolor": "white",
             "linecolor": "white",
             "minorgridcolor": "white",
             "startlinecolor": "#2a3f5f"
            },
            "baxis": {
             "endlinecolor": "#2a3f5f",
             "gridcolor": "white",
             "linecolor": "white",
             "minorgridcolor": "white",
             "startlinecolor": "#2a3f5f"
            },
            "type": "carpet"
           }
          ],
          "choropleth": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "type": "choropleth"
           }
          ],
          "contour": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "contour"
           }
          ],
          "contourcarpet": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "type": "contourcarpet"
           }
          ],
          "heatmap": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "heatmap"
           }
          ],
          "heatmapgl": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "heatmapgl"
           }
          ],
          "histogram": [
           {
            "marker": {
             "pattern": {
              "fillmode": "overlay",
              "size": 10,
              "solidity": 0.2
             }
            },
            "type": "histogram"
           }
          ],
          "histogram2d": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "histogram2d"
           }
          ],
          "histogram2dcontour": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "histogram2dcontour"
           }
          ],
          "mesh3d": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "type": "mesh3d"
           }
          ],
          "parcoords": [
           {
            "line": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "parcoords"
           }
          ],
          "pie": [
           {
            "automargin": true,
            "type": "pie"
           }
          ],
          "scatter": [
           {
            "fillpattern": {
             "fillmode": "overlay",
             "size": 10,
             "solidity": 0.2
            },
            "type": "scatter"
           }
          ],
          "scatter3d": [
           {
            "line": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatter3d"
           }
          ],
          "scattercarpet": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattercarpet"
           }
          ],
          "scattergeo": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattergeo"
           }
          ],
          "scattergl": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattergl"
           }
          ],
          "scattermapbox": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattermapbox"
           }
          ],
          "scatterpolar": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatterpolar"
           }
          ],
          "scatterpolargl": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatterpolargl"
           }
          ],
          "scatterternary": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatterternary"
           }
          ],
          "surface": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "surface"
           }
          ],
          "table": [
           {
            "cells": {
             "fill": {
              "color": "#EBF0F8"
             },
             "line": {
              "color": "white"
             }
            },
            "header": {
             "fill": {
              "color": "#C8D4E3"
             },
             "line": {
              "color": "white"
             }
            },
            "type": "table"
           }
          ]
         },
         "layout": {
          "annotationdefaults": {
           "arrowcolor": "#2a3f5f",
           "arrowhead": 0,
           "arrowwidth": 1
          },
          "autotypenumbers": "strict",
          "coloraxis": {
           "colorbar": {
            "outlinewidth": 0,
            "ticks": ""
           }
          },
          "colorscale": {
           "diverging": [
            [
             0,
             "#8e0152"
            ],
            [
             0.1,
             "#c51b7d"
            ],
            [
             0.2,
             "#de77ae"
            ],
            [
             0.3,
             "#f1b6da"
            ],
            [
             0.4,
             "#fde0ef"
            ],
            [
             0.5,
             "#f7f7f7"
            ],
            [
             0.6,
             "#e6f5d0"
            ],
            [
             0.7,
             "#b8e186"
            ],
            [
             0.8,
             "#7fbc41"
            ],
            [
             0.9,
             "#4d9221"
            ],
            [
             1,
             "#276419"
            ]
           ],
           "sequential": [
            [
             0,
             "#0d0887"
            ],
            [
             0.1111111111111111,
             "#46039f"
            ],
            [
             0.2222222222222222,
             "#7201a8"
            ],
            [
             0.3333333333333333,
             "#9c179e"
            ],
            [
             0.4444444444444444,
             "#bd3786"
            ],
            [
             0.5555555555555556,
             "#d8576b"
            ],
            [
             0.6666666666666666,
             "#ed7953"
            ],
            [
             0.7777777777777778,
             "#fb9f3a"
            ],
            [
             0.8888888888888888,
             "#fdca26"
            ],
            [
             1,
             "#f0f921"
            ]
           ],
           "sequentialminus": [
            [
             0,
             "#0d0887"
            ],
            [
             0.1111111111111111,
             "#46039f"
            ],
            [
             0.2222222222222222,
             "#7201a8"
            ],
            [
             0.3333333333333333,
             "#9c179e"
            ],
            [
             0.4444444444444444,
             "#bd3786"
            ],
            [
             0.5555555555555556,
             "#d8576b"
            ],
            [
             0.6666666666666666,
             "#ed7953"
            ],
            [
             0.7777777777777778,
             "#fb9f3a"
            ],
            [
             0.8888888888888888,
             "#fdca26"
            ],
            [
             1,
             "#f0f921"
            ]
           ]
          },
          "colorway": [
           "#636efa",
           "#EF553B",
           "#00cc96",
           "#ab63fa",
           "#FFA15A",
           "#19d3f3",
           "#FF6692",
           "#B6E880",
           "#FF97FF",
           "#FECB52"
          ],
          "font": {
           "color": "#2a3f5f"
          },
          "geo": {
           "bgcolor": "white",
           "lakecolor": "white",
           "landcolor": "#E5ECF6",
           "showlakes": true,
           "showland": true,
           "subunitcolor": "white"
          },
          "hoverlabel": {
           "align": "left"
          },
          "hovermode": "closest",
          "mapbox": {
           "style": "light"
          },
          "paper_bgcolor": "white",
          "plot_bgcolor": "#E5ECF6",
          "polar": {
           "angularaxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           },
           "bgcolor": "#E5ECF6",
           "radialaxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           }
          },
          "scene": {
           "xaxis": {
            "backgroundcolor": "#E5ECF6",
            "gridcolor": "white",
            "gridwidth": 2,
            "linecolor": "white",
            "showbackground": true,
            "ticks": "",
            "zerolinecolor": "white"
           },
           "yaxis": {
            "backgroundcolor": "#E5ECF6",
            "gridcolor": "white",
            "gridwidth": 2,
            "linecolor": "white",
            "showbackground": true,
            "ticks": "",
            "zerolinecolor": "white"
           },
           "zaxis": {
            "backgroundcolor": "#E5ECF6",
            "gridcolor": "white",
            "gridwidth": 2,
            "linecolor": "white",
            "showbackground": true,
            "ticks": "",
            "zerolinecolor": "white"
           }
          },
          "shapedefaults": {
           "line": {
            "color": "#2a3f5f"
           }
          },
          "ternary": {
           "aaxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           },
           "baxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           },
           "bgcolor": "#E5ECF6",
           "caxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           }
          },
          "title": {
           "x": 0.05
          },
          "xaxis": {
           "automargin": true,
           "gridcolor": "white",
           "linecolor": "white",
           "ticks": "",
           "title": {
            "standoff": 15
           },
           "zerolinecolor": "white",
           "zerolinewidth": 2
          },
          "yaxis": {
           "automargin": true,
           "gridcolor": "white",
           "linecolor": "white",
           "ticks": "",
           "title": {
            "standoff": 15
           },
           "zerolinecolor": "white",
           "zerolinewidth": 2
          }
         }
        },
        "title": {
         "text": "Crime Counts by Sector and Category"
        },
        "width": 2000,
        "xaxis": {
         "categoryorder": "total descending",
         "title": {
          "text": "Sector"
         }
        },
        "yaxis": {
         "title": {
          "text": "Crime Counts"
         }
        }
       }
      }
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "import pandas as pd\n",
    "import plotly.graph_objects as go\n",
    "\n",
    "pivot_data = df_clean.pivot_table(index='Sector', columns='Category', values='Crime Count', aggfunc='sum', fill_value=0)\n",
    "categories = pivot_data.columns\n",
    "sectors = pivot_data.index\n",
    "data_for_plotting = []\n",
    "\n",
    "for category in categories:\n",
    "    data_for_plotting.append(\n",
    "        go.Bar(\n",
    "            name=category,\n",
    "            x=sectors,\n",
    "            y=pivot_data[category],\n",
    "            marker=dict(\n",
    "                line=dict(color='rgb(248, 248, 249)', width=1)\n",
    "            )\n",
    "        )\n",
    "    )\n",
    "\n",
    "fig = go.Figure(data=data_for_plotting)\n",
    "fig.update_layout(\n",
    "    barmode='stack',\n",
    "    xaxis={'categoryorder': 'total descending'},\n",
    "    title='Crime Counts by Sector and Category',\n",
    "    xaxis_title=\"Sector\",\n",
    "    yaxis_title=\"Crime Counts\",\n",
    "    legend_title=\"Category\",\n",
    "    plot_bgcolor='rgba(0,0,0,0)',\n",
    "    width=2000,  \n",
    "    height=1000\n",
    ")\n",
    "\n",
    "fig.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "From this charts, we can see that thw top 2 sectors ith most crimes are 'Centre' and 'Northwest', with 'Theft from auto' being the most current crime in both locations."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Now I am going to plot an interactive geo spacial chart to see this in a more visual way"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Geo Spatial"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/vnd.plotly.v1+json": {
       "config": {
        "plotlyServerURL": "https://plot.ly"
       },
       "data": [
        {
         "hovertemplate": "<b>%{hovertext}</b><br><br>Latitude=%{lat}<br>Longitude=%{lon}<br>Crime Count=%{marker.color}<extra></extra>",
         "hovertext": [
          "01B",
          "01F",
          "01K",
          "02B",
          "02C",
          "02E",
          "02F",
          "02K",
          "03W",
          "05D",
          "05F",
          "06A",
          "06C",
          "09H",
          "09O",
          "09P",
          "09Q",
          "10D",
          "10E",
          "12A",
          "12B",
          "12C",
          "12J",
          "12K",
          "12L",
          "13A",
          "13E",
          "13F",
          "13G",
          "13M",
          "ABBEYDALE",
          "ACADIA",
          "ALBERT PARK/RADISSON HEIGHTS",
          "ALTADORE",
          "ALYTH/BONNYBROOK",
          "APPLEWOOD PARK",
          "ARBOUR LAKE",
          "ASPEN WOODS",
          "AUBURN BAY",
          "AURORA BUSINESS PARK",
          "BANFF TRAIL",
          "BANKVIEW",
          "BAYVIEW",
          "BEDDINGTON HEIGHTS",
          "BEL-AIRE",
          "BELMONT",
          "BELTLINE",
          "BELVEDERE",
          "BONAVISTA DOWNS",
          "BOWNESS",
          "BRAESIDE",
          "BRENTWOOD",
          "BRIDGELAND/RIVERSIDE",
          "BRIDLEWOOD",
          "BRITANNIA",
          "BURNS INDUSTRIAL",
          "CALGARY INTERNATIONAL AIRPORT",
          "CAMBRIAN HEIGHTS",
          "CANADA OLYMPIC PARK",
          "CANYON MEADOWS",
          "CAPITOL HILL",
          "CARRINGTON",
          "CASTLERIDGE",
          "CEDARBRAE",
          "CHAPARRAL",
          "CHARLESWOOD",
          "CHINATOWN",
          "CHINOOK PARK",
          "CHRISTIE PARK",
          "CITADEL",
          "CITYSCAPE",
          "CLIFF BUNGALOW",
          "COACH HILL",
          "COLLINGWOOD",
          "COPPERFIELD",
          "CORAL SPRINGS",
          "CORNERSTONE",
          "COUGAR RIDGE",
          "COUNTRY HILLS",
          "COUNTRY HILLS VILLAGE",
          "COVENTRY HILLS",
          "CRANSTON",
          "CRESCENT HEIGHTS",
          "CRESTMONT",
          "CURRIE BARRACKS",
          "DALHOUSIE",
          "DEER RIDGE",
          "DEER RUN",
          "DEERFOOT BUSINESS CENTRE",
          "DIAMOND COVE",
          "DISCOVERY RIDGE",
          "DOUGLASDALE/GLEN",
          "DOVER",
          "DOWNTOWN COMMERCIAL CORE",
          "DOWNTOWN EAST VILLAGE",
          "DOWNTOWN WEST END",
          "EAGLE RIDGE",
          "EAST FAIRVIEW INDUSTRIAL",
          "EAST SHEPARD INDUSTRIAL",
          "EASTFIELD",
          "EAU CLAIRE",
          "EDGEMONT",
          "ELBOW PARK",
          "ELBOYA",
          "ERIN WOODS",
          "ERLTON",
          "EVANSTON",
          "EVERGREEN",
          "FAIRVIEW",
          "FAIRVIEW INDUSTRIAL",
          "FALCONRIDGE",
          "FISH CREEK PARK",
          "FOOTHILLS",
          "FOREST HEIGHTS",
          "FOREST LAWN",
          "FOREST LAWN INDUSTRIAL",
          "FRANKLIN",
          "GARRISON GREEN",
          "GARRISON WOODS",
          "GLAMORGAN",
          "GLENBROOK",
          "GLENDALE",
          "GLENDEER BUSINESS PARK",
          "GLENMORE PARK",
          "GOLDEN TRIANGLE",
          "GREAT PLAINS",
          "GREENVIEW",
          "GREENVIEW INDUSTRIAL PARK",
          "GREENWOOD/GREENBRIAR",
          "HAMPTONS",
          "HARVEST HILLS",
          "HASKAYNE",
          "HAWKWOOD",
          "HAYSBORO",
          "HIDDEN VALLEY",
          "HIGHFIELD",
          "HIGHLAND PARK",
          "HIGHWOOD",
          "HILLHURST",
          "HOMESTEAD",
          "HORIZON",
          "HOTCHKISS",
          "HOUNSFIELD HEIGHTS/BRIAR HILL",
          "HUNTINGTON HILLS",
          "INGLEWOOD",
          "KELVIN GROVE",
          "KILLARNEY/GLENGARRY",
          "KINCORA",
          "KINGSLAND",
          "LAKE BONAVISTA",
          "LAKEVIEW",
          "LEGACY",
          "LINCOLN PARK",
          "LIVINGSTON",
          "LOWER MOUNT ROYAL",
          "MACEWAN GLEN",
          "MAHOGANY",
          "MANCHESTER",
          "MANCHESTER INDUSTRIAL",
          "MAPLE RIDGE",
          "MARLBOROUGH",
          "MARLBOROUGH PARK",
          "MARTINDALE",
          "MAYFAIR",
          "MAYLAND",
          "MAYLAND HEIGHTS",
          "MCCALL",
          "MCKENZIE LAKE",
          "MCKENZIE TOWNE",
          "MEADOWLARK PARK",
          "MEDICINE HILL",
          "MERIDIAN",
          "MIDNAPORE",
          "MILLRISE",
          "MISSION",
          "MONTEREY PARK",
          "MONTGOMERY",
          "MOUNT PLEASANT",
          "NEW BRIGHTON",
          "NOLAN HILL",
          "NORTH AIRWAYS",
          "NORTH GLENMORE PARK",
          "NORTH HAVEN",
          "NORTH HAVEN UPPER",
          "NOSE HILL PARK",
          "OAKRIDGE",
          "OGDEN",
          "OGDEN SHOPS",
          "PALLISER",
          "PANORAMA HILLS",
          "PARKDALE",
          "PARKHILL",
          "PARKLAND",
          "PATTERSON",
          "PEGASUS",
          "PENBROOKE MEADOWS",
          "PINE CREEK",
          "PINERIDGE",
          "POINT MCKAY",
          "PUMP HILL",
          "QUEENS PARK VILLAGE",
          "QUEENSLAND",
          "RAMSAY",
          "RANCHLANDS",
          "RANGEVIEW",
          "RED CARPET",
          "REDSTONE",
          "RENFREW",
          "RICHMOND",
          "RIDEAU PARK",
          "RIVERBEND",
          "ROCKY RIDGE",
          "ROSEDALE",
          "ROSEMONT",
          "ROSSCARROCK",
          "ROXBORO",
          "ROYAL OAK",
          "ROYAL VISTA",
          "RUNDLE",
          "RUTLAND PARK",
          "SADDLE RIDGE",
          "SADDLE RIDGE INDUSTRIAL",
          "SAGE HILL",
          "SANDSTONE VALLEY",
          "SCARBORO",
          "SCARBORO/ SUNALTA WEST",
          "SCENIC ACRES",
          "SECTION 23",
          "SETON",
          "SHAGANAPPI",
          "SHAWNEE SLOPES",
          "SHAWNESSY",
          "SHEPARD INDUSTRIAL",
          "SHERWOOD",
          "SIGNAL HILL",
          "SILVER SPRINGS",
          "SILVERADO",
          "SKYLINE EAST",
          "SKYLINE WEST",
          "SKYVIEW RANCH",
          "SOMERSET",
          "SOUTH AIRWAYS",
          "SOUTH CALGARY",
          "SOUTH FOOTHILLS",
          "SOUTHVIEW",
          "SOUTHWOOD",
          "SPRINGBANK HILL",
          "SPRUCE CLIFF",
          "ST. ANDREWS HEIGHTS",
          "STARFIELD",
          "STONEGATE LANDING",
          "STONEY 1",
          "STONEY 2",
          "STONEY 3",
          "STONEY 4",
          "STRATHCONA PARK",
          "SUNALTA",
          "SUNDANCE",
          "SUNNYSIDE",
          "SUNRIDGE",
          "TARADALE",
          "TEMPLE",
          "THORNCLIFFE",
          "TUSCANY",
          "TUXEDO PARK",
          "UNIVERSITY DISTRICT",
          "UNIVERSITY HEIGHTS",
          "UNIVERSITY OF CALGARY",
          "UPPER MOUNT ROYAL",
          "VALLEY RIDGE",
          "VALLEYFIELD",
          "VARSITY",
          "VISTA HEIGHTS",
          "WALDEN",
          "WEST HILLHURST",
          "WEST SPRINGS",
          "WESTGATE",
          "WESTWINDS",
          "WHITEHORN",
          "WILDWOOD",
          "WILLOW PARK",
          "WINDSOR PARK",
          "WINSTON HEIGHTS/MOUNTVIEW",
          "WOLF WILLOW",
          "WOODBINE",
          "WOODLANDS",
          "YORKVILLE"
         ],
         "lat": [
          51.10282600762854,
          51.11734815371878,
          51.16872429008923,
          51.1760213513024,
          51.175901183468724,
          51.1614306187993,
          51.1597635370127,
          51.19055262524679,
          51.197967711164964,
          51.179594126023254,
          51.147035235570414,
          51.05901467759118,
          51.08429730555001,
          51.00636522980599,
          51.05257108080651,
          51.02684760243017,
          51.026252049821956,
          51.103350599050756,
          51.081453566081535,
          50.96505615111287,
          50.93922077318547,
          50.925106282431614,
          50.88358305143726,
          50.975824539924844,
          50.914350106443706,
          50.914133045147565,
          50.8993917852468,
          50.89941005804681,
          50.8958227049736,
          50.85623560571985,
          51.05941500696419,
          50.97241507334731,
          51.044532952523866,
          51.015954118298694,
          51.02285923852866,
          51.0448748321394,
          51.1325947114686,
          51.04519167765947,
          50.89270203779492,
          51.136262432235235,
          51.07421633024228,
          51.0341275275111,
          50.97463475577881,
          51.13164398532214,
          50.99944254944499,
          50.86872241539765,
          51.037437963359764,
          51.037827470246434,
          50.94373042986891,
          51.0833462877166,
          50.95600269125181,
          51.096065536203874,
          51.05055069436678,
          50.89924691819346,
          51.01264400380039,
          50.99870693503925,
          51.123556453987604,
          51.08433917204962,
          51.08123298660904,
          50.94095720863543,
          51.07266527843247,
          51.18568826121794,
          51.105254134534725,
          50.958199454986094,
          50.88467246113951,
          51.090088050163416,
          51.05282359454732,
          50.98322809371844,
          51.04079952918998,
          51.14453875913651,
          51.14703155696725,
          51.033933194742325,
          51.05838582508582,
          51.08350990566399,
          50.91781309734389,
          51.103352770689,
          51.157359967042765,
          51.0718808513961,
          51.1437756831102,
          51.158203848304325,
          51.16490633029946,
          50.87819972206263,
          51.05949594683074,
          51.083702771535876,
          51.01572348884976,
          51.109409937360724,
          50.926573869162,
          50.92540672204504,
          51.11209775483432,
          50.94621275230468,
          51.01566153397868,
          50.94844437195182,
          51.02256772250409,
          51.047274435217574,
          51.04709033069666,
          51.047969140191825,
          50.988744274044464,
          50.98403287936156,
          50.95130983742685,
          51.01194930387265,
          51.053606993489936,
          51.12458784624158,
          51.02134304173559,
          51.012993345448066,
          51.0209762755992,
          51.026333793586,
          51.171085356450234,
          50.91871688494811,
          50.985382121323525,
          50.987010064019884,
          51.10330638378534,
          50.90960459313166,
          50.99396302917181,
          51.049243148110726,
          51.03784804797574,
          51.02709477694866,
          51.058927589452686,
          51.003055154177616,
          51.01717823051979,
          51.0154042701951,
          51.02419405241394,
          51.033371391768654,
          50.9900931253413,
          50.99046841956801,
          51.01339075567755,
          50.98689494676143,
          51.09485613506574,
          51.08651820065337,
          51.08986537100539,
          51.14509283969437,
          51.14767758603047,
          51.11697203613649,
          51.13097810152047,
          50.972214842335006,
          51.15193708420534,
          51.01852460881388,
          51.088018178558066,
          51.092359423036356,
          51.05727753172276,
          51.121629709124846,
          51.08990091526667,
          50.91420100643065,
          51.06317953856462,
          51.11758229183931,
          51.034140523504476,
          50.989881144871724,
          51.03013195588548,
          51.16026670748173,
          50.98697674970066,
          50.939803390308995,
          50.99957240837904,
          50.85761440424294,
          51.01027374327917,
          51.185504064958366,
          51.0365031977319,
          51.13771903639299,
          50.89588555708811,
          51.00624267544188,
          51.013255989792455,
          50.957637582999936,
          51.05954228117922,
          51.059688495333184,
          51.11791372594209,
          50.9958120430023,
          51.05239255688014,
          51.0608334639511,
          51.08899753775383,
          50.91487440011554,
          50.91714622424966,
          50.997857843950264,
          51.07918367485378,
          51.0565835816587,
          50.915593553179335,
          50.91704529217144,
          51.03315775861924,
          51.08145439005355,
          51.071076143066776,
          51.074237908423605,
          50.92206464811321,
          51.175293692964885,
          51.08713865081421,
          51.00142837215398,
          51.095941551559434,
          51.10440179087063,
          51.11127714093448,
          50.96966279895669,
          50.991878337981746,
          50.99108924416474,
          50.96676703111173,
          51.15971101107547,
          51.05994755064559,
          51.018192290150814,
          50.92120040109183,
          51.06393987638284,
          51.09447884904051,
          51.04585328763989,
          50.85626936158896,
          51.074259757631374,
          51.05954116603247,
          50.966827701378456,
          51.08514445824785,
          50.938224120412734,
          51.03641206367581,
          51.11917908233608,
          50.87437929445921,
          51.04258062657982,
          51.170446168252354,
          51.059653157676735,
          51.02971029482691,
          51.02487576913786,
          50.976018859070834,
          51.14407647895953,
          51.062686663226856,
          51.078394197521575,
          51.04328787137517,
          51.02619585231578,
          51.142660730912894,
          51.149421809959804,
          51.074239973666835,
          51.018990986828264,
          51.13057324213389,
          51.125189506147144,
          51.175616972779984,
          51.1377328448926,
          51.04076552661671,
          51.042200316401306,
          51.11270750759221,
          50.97092731249437,
          50.87413786105747,
          51.04521136346261,
          50.92618889834688,
          50.90337065586358,
          50.96149445512731,
          51.159990919144846,
          51.023013957074795,
          51.10427454305881,
          50.88298879992247,
          51.100152612629664,
          51.103332199733316,
          51.15612673497898,
          50.89891142227545,
          51.0742191886852,
          51.02680215136445,
          50.97124216017267,
          51.03415221260387,
          50.957758807711244,
          51.026602437834555,
          51.048659707650735,
          51.06391055940229,
          51.00485589414011,
          51.16446858199437,
          51.16118788060755,
          51.14338378225708,
          51.14703268582613,
          51.17620448471432,
          51.047410137464645,
          51.04295989234576,
          50.90164564994187,
          51.05641953275016,
          51.074221599518935,
          51.11794016304313,
          51.08879458506016,
          51.10310302976012,
          51.12191108762252,
          51.07421777158121,
          51.07385411760001,
          51.07040418011127,
          51.075020786453926,
          51.02997367628775,
          51.095388615664305,
          51.009084099675064,
          51.096053770173505,
          51.07193318391342,
          50.870406291791525,
          51.05507951363527,
          51.05977978013603,
          51.044177846119624,
          51.10339241438977,
          51.088779871474486,
          51.05825824761795,
          50.95662379617276,
          51.00504544398754,
          51.07529880217513,
          50.870710533378784,
          50.94023326198639,
          50.94112397805165,
          50.870356694433404
         ],
         "legendgroup": "",
         "lon": [
          -114.24261426059833,
          -114.2611864663544,
          -114.22271630893266,
          -114.19939956642429,
          -114.17664928905774,
          -114.19943706682423,
          -114.17512030034136,
          -114.19945286812306,
          -114.0262426340911,
          -113.95866084614936,
          -113.91649823812915,
          -114.22966502126448,
          -114.22745425562907,
          -114.01538929528147,
          -113.87710694134275,
          -113.88876476378506,
          -113.93396103455751,
          -113.91603045928909,
          -113.91623374150298,
          -113.89323152147792,
          -113.8986612080914,
          -113.90342610328828,
          -113.88348471893552,
          -113.91291790161674,
          -113.87152264476616,
          -114.1981928602642,
          -114.19822810239104,
          -114.17505002464321,
          -114.14347009888938,
          -114.04275071508103,
          -113.92780385615137,
          -114.05369601618155,
          -113.9967779352673,
          -114.10078239468565,
          -114.02248221491362,
          -113.92781025877834,
          -114.20767498075155,
          -114.20797769742218,
          -113.9578101488218,
          -114.05847554152582,
          -114.11512839716917,
          -114.10048318978971,
          -114.11396431346715,
          -114.08491114080482,
          -114.08893425156596,
          -114.05528001525202,
          -114.05518608886146,
          -113.9049830348633,
          -114.03150547303466,
          -114.18838847472048,
          -114.10636590290262,
          -114.1359191739885,
          -114.03507983816583,
          -114.11022194804715,
          -114.08627738153702,
          -114.03966866537291,
          -114.01171665690258,
          -114.08937896523537,
          -114.21670015081224,
          -114.08200634588414,
          -114.09463117970881,
          -114.08876544116907,
          -113.96102973195798,
          -114.12962160153805,
          -114.02566876175794,
          -114.1122123154022,
          -114.06552190579208,
          -114.0889939534622,
          -114.17601756827449,
          -114.18393693294375,
          -113.96125803148381,
          -114.07490165596653,
          -114.18297960518734,
          -114.1024036222742,
          -113.92946741896215,
          -113.92694848647866,
          -113.93057052562796,
          -114.2091625968558,
          -114.0798937747985,
          -114.06139543185002,
          -114.05314340697439,
          -113.97999557792768,
          -114.06161323521818,
          -114.26905946198636,
          -114.1248177444654,
          -114.15815535928421,
          -114.02023404987288,
          -114.00908770459678,
          -114.0383269425689,
          -114.02265590844065,
          -114.21710362789646,
          -114.01190805579242,
          -113.99305400906283,
          -114.06959371115288,
          -114.05084794560608,
          -114.08777935920926,
          -114.09679322093368,
          -114.03863977769701,
          -113.9509699643221,
          -113.97004371849067,
          -114.07282181932268,
          -114.14395321026961,
          -114.083805311106,
          -114.07360459021476,
          -113.96766881721656,
          -114.06314648304446,
          -114.11245112799257,
          -114.1156755778959,
          -114.06036094398164,
          -114.06878074585325,
          -113.94537499024918,
          -114.02652359670793,
          -113.98203822530505,
          -113.96875793277411,
          -113.97000191799683,
          -113.95696985982971,
          -113.98752400359126,
          -114.12452628199492,
          -114.11361449884893,
          -114.15322753874968,
          -114.15269483793375,
          -114.15271889005692,
          -114.03196872405518,
          -114.13153370711461,
          -113.9853085392185,
          -113.9399030804802,
          -114.05746990262463,
          -114.04905866998868,
          -114.22297068115356,
          -114.14668419231347,
          -114.05277797732376,
          -114.2866952846797,
          -114.17561536315762,
          -114.08334023954198,
          -114.11960923256242,
          -114.037193826917,
          -114.0620558477431,
          -114.07989112691597,
          -114.09364601408916,
          -113.91650032535958,
          -113.99169780219123,
          -113.8962748920383,
          -114.10572976665989,
          -114.06670944395931,
          -114.01912581291103,
          -114.08902826660753,
          -114.13173134426289,
          -114.13286512975175,
          -114.07735883671273,
          -114.05133665127042,
          -114.1296213924465,
          -114.00599781298918,
          -114.12999189961174,
          -114.06550670687392,
          -114.08649939756594,
          -114.11521349389004,
          -113.92455389137416,
          -114.0666537824219,
          -114.05722246720464,
          -114.0346890274,
          -113.96998768840771,
          -113.94671708302872,
          -113.9583871171306,
          -114.08895573448329,
          -114.01030237788878,
          -114.01331470323042,
          -114.02708667609807,
          -113.98818319020536,
          -113.96143747927303,
          -114.07735167071722,
          -114.20014671121054,
          -113.99651752174083,
          -114.05760395145887,
          -114.08490366384473,
          -114.06808259035802,
          -113.92685104381658,
          -114.16224821352257,
          -114.07602998447108,
          -113.94524646254676,
          -114.1663172033934,
          -114.00752398291431,
          -114.11225607544628,
          -114.08942057511393,
          -114.08535430355379,
          -114.11069096474999,
          -114.12961375709601,
          -114.01244227109757,
          -113.99316026930774,
          -114.11215889038718,
          -114.08806340626337,
          -114.13533325689279,
          -114.06553452471233,
          -114.02854335688808,
          -114.1771423135999,
          -114.00648886102513,
          -113.94768720611339,
          -114.06787772053065,
          -113.94676495577161,
          -114.14614868218266,
          -114.10078728034952,
          -114.0776067500382,
          -114.0225717060612,
          -114.04253229079222,
          -114.1836469813454,
          -113.9197443693882,
          -113.93892582460018,
          -113.95750282716884,
          -114.03916899334408,
          -114.11842028929458,
          -114.07416183074736,
          -114.01843029189821,
          -114.24590791801509,
          -114.07820713414624,
          -114.09003307555233,
          -114.1454859092963,
          -114.06764480275648,
          -114.21976187387784,
          -114.20423704320459,
          -113.97005811864892,
          -114.13736058542634,
          -113.94562252266931,
          -113.97572968236095,
          -114.14068609335015,
          -114.09496250976525,
          -114.10717977139831,
          -114.11488200443591,
          -114.2173409568748,
          -113.94813250933105,
          -113.94791315312588,
          -114.12665233527957,
          -114.08580303773778,
          -114.07392552001497,
          -113.99278635609633,
          -114.1520001467928,
          -114.1799070291585,
          -114.19628181361762,
          -114.07129778862543,
          -114.03893128621466,
          -114.04740128490002,
          -113.95837456074635,
          -114.08138084069279,
          -114.0052956757618,
          -114.10207470148995,
          -113.97367259111124,
          -113.99733916298928,
          -114.08319427981235,
          -114.20885770724253,
          -114.13627783911859,
          -114.12743730040928,
          -113.94002380686328,
          -113.98668274803757,
          -114.02469848732257,
          -114.01721821479654,
          -113.97583513108833,
          -114.0027575486726,
          -114.17758676157283,
          -114.10148594325165,
          -114.04206714385867,
          -114.07902910930551,
          -113.99170056999279,
          -113.93400207797252,
          -113.94676773665904,
          -114.06878043292588,
          -114.24088620535427,
          -114.06104493215598,
          -114.14496478086903,
          -114.13308657031214,
          -114.12993378392255,
          -114.0850256868341,
          -114.25129737820474,
          -114.00188586057929,
          -114.1625067595007,
          -114.01770522719029,
          -114.03261949596039,
          -114.11515850909333,
          -114.20633960635404,
          -114.15859390257255,
          -113.97005363203272,
          -113.9700681964178,
          -114.15853861691606,
          -114.0561955045225,
          -114.08355224178209,
          -114.04184874950579,
          -114.00862958290263,
          -114.1296281727553,
          -114.1061595815691,
          -114.08234638755263
         ],
         "marker": {
          "color": [
           25,
           11,
           73,
           3,
           19,
           415,
           22,
           4,
           15,
           20,
           2,
           10,
           2,
           7,
           92,
           12,
           50,
           18,
           107,
           317,
           23,
           6,
           83,
           2,
           29,
           2,
           1,
           26,
           18,
           7,
           665,
           1860,
           1923,
           1019,
           555,
           671,
           1303,
           514,
           924,
           99,
           1267,
           1433,
           97,
           1170,
           54,
           27,
           9471,
           153,
           59,
           2429,
           520,
           1444,
           2035,
           547,
           132,
           417,
           957,
           278,
           42,
           853,
           1299,
           125,
           884,
           455,
           524,
           405,
           425,
           185,
           189,
           375,
           262,
           901,
           231,
           320,
           691,
           450,
           425,
           334,
           301,
           572,
           1018,
           885,
           2114,
           108,
           36,
           994,
           364,
           265,
           302,
           50,
           223,
           896,
           2113,
           6876,
           1870,
           596,
           227,
           467,
           979,
           244,
           546,
           783,
           443,
           351,
           965,
           338,
           666,
           727,
           621,
           614,
           1952,
           109,
           1538,
           1271,
           2996,
           556,
           958,
           145,
           125,
           829,
           877,
           529,
           57,
           203,
           164,
           296,
           435,
           895,
           82,
           299,
           718,
           23,
           466,
           1422,
           597,
           764,
           1060,
           496,
           1996,
           17,
           1222,
           12,
           1030,
           1874,
           1098,
           344,
           1475,
           370,
           1309,
           1146,
           505,
           477,
           308,
           169,
           1019,
           264,
           630,
           649,
           2315,
           214,
           2210,
           1416,
           1760,
           54,
           367,
           1223,
           461,
           698,
           1283,
           907,
           23,
           873,
           647,
           489,
           1322,
           858,
           1409,
           1447,
           611,
           332,
           584,
           385,
           220,
           44,
           170,
           420,
           1280,
           42,
           330,
           1257,
           397,
           521,
           207,
           332,
           139,
           1511,
           28,
           2029,
           108,
           138,
           66,
           340,
           444,
           567,
           31,
           239,
           427,
           1561,
           1016,
           97,
           694,
           342,
           259,
           183,
           1111,
           54,
           508,
           171,
           2029,
           321,
           1946,
           284,
           632,
           329,
           184,
           59,
           503,
           115,
           562,
           442,
           291,
           1274,
           499,
           431,
           1205,
           696,
           354,
           123,
           218,
           1207,
           601,
           974,
           1104,
           348,
           734,
           1359,
           559,
           502,
           570,
           207,
           132,
           187,
           710,
           206,
           99,
           373,
           1329,
           521,
           1127,
           2094,
           1511,
           1677,
           1417,
           800,
           1586,
           180,
           416,
           142,
           404,
           281,
           318,
           1770,
           569,
           364,
           1210,
           580,
           300,
           1101,
           1626,
           317,
           1109,
           1052,
           972,
           21,
           540,
           583,
           25
          ],
          "coloraxis": "coloraxis",
          "colorscale": [
           [
            0,
            "#440154"
           ],
           [
            0.1111111111111111,
            "#482878"
           ],
           [
            0.2222222222222222,
            "#3e4989"
           ],
           [
            0.3333333333333333,
            "#31688e"
           ],
           [
            0.4444444444444444,
            "#26828e"
           ],
           [
            0.5555555555555556,
            "#1f9e89"
           ],
           [
            0.6666666666666666,
            "#35b779"
           ],
           [
            0.7777777777777778,
            "#6ece58"
           ],
           [
            0.8888888888888888,
            "#b5de2b"
           ],
           [
            1,
            "#fde725"
           ]
          ],
          "showscale": true,
          "size": [
           0.15837820715869497,
           0.06968641114982578,
           0.46246436490338927,
           0.019005384859043396,
           0.12036743744060817,
           2.6290782388343366,
           0.13937282229965156,
           0.025340513145391194,
           0.09502692429521697,
           0.12670256572695598,
           0.012670256572695597,
           0.06335128286347799,
           0.012670256572695597,
           0.044345898004434586,
           0.5828318023439975,
           0.07602153943617358,
           0.31675641431738993,
           0.11403230915426038,
           0.6778587266392145,
           2.0082356667722525,
           0.14570795058599936,
           0.03801076971808679,
           0.5258156477668673,
           0.012670256572695597,
           0.18371872030408615,
           0.012670256572695597,
           0.0063351282863477985,
           0.16471333544504277,
           0.11403230915426038,
           0.044345898004434586,
           4.212860310421286,
           11.783338612606904,
           12.182451694646817,
           6.455495723788407,
           3.515996198923028,
           4.2508710801393725,
           8.254672157111182,
           3.2562559391827683,
           5.853658536585366,
           0.6271777003484321,
           8.026607538802661,
           9.078238834336396,
           0.6145074437757365,
           7.412100095026924,
           0.34209692746278114,
           0.17104846373139057,
           60,
           0.9692746278112131,
           0.3737725688945201,
           15.388026607538803,
           3.2942667089008553,
           9.14792524548622,
           12.89198606271777,
           3.4653151726322458,
           0.8362369337979094,
           2.641748495407032,
           6.062717770034843,
           1.7611656636046882,
           0.2660753880266075,
           5.403864428254672,
           8.229331643965791,
           0.7918910357934749,
           5.600253405131454,
           2.882483370288248,
           3.3196072220462467,
           2.5657269559708586,
           2.6924295216978145,
           1.1719987329743426,
           1.1973392461197339,
           2.3756731073804245,
           1.6598036110231233,
           5.707950585999367,
           1.4634146341463414,
           2.0272410516312958,
           4.377573645866328,
           2.850807728856509,
           2.6924295216978145,
           2.1159328476401646,
           1.9068736141906875,
           3.623693379790941,
           6.449160595502059,
           5.606588533417802,
           13.392461197339246,
           0.6841938549255623,
           0.22806461830852076,
           6.2971175166297115,
           2.305986696230599,
           1.6788089958821666,
           1.9132087424770352,
           0.31675641431738993,
           1.412733607855559,
           5.676274944567627,
           13.386126069052898,
           43.560342096927464,
           11.846689895470382,
           3.7757364586632876,
           1.4380741210009502,
           2.9585049097244216,
           6.2020905923344944,
           1.5457713018688628,
           3.458980044345898,
           4.960405448210326,
           2.806461830852075,
           2.2236300285080772,
           6.113398796325625,
           2.1412733607855556,
           4.219195438707634,
           4.605638264174849,
           3.934114665821983,
           3.889768767817548,
           12.366170414950902,
           0.6905289832119101,
           9.743427304402914,
           8.051948051948052,
           18.980044345898005,
           3.5223313272093764,
           6.069052898321191,
           0.9185936015204308,
           0.7918910357934749,
           5.251821349382325,
           5.55590750712702,
           3.3512828634779854,
           0.3611023123218245,
           1.2860310421286032,
           1.0389610389610389,
           1.8751979727589483,
           2.7557808045612924,
           5.66993981628128,
           0.5194805194805194,
           1.8942033576179915,
           4.548622109597719,
           0.14570795058599936,
           2.952169781438074,
           9.008552423186568,
           3.7820715869496353,
           4.840038010769718,
           6.715235983528666,
           3.142223630028508,
           12.644916059550205,
           0.10769718086791258,
           7.741526765917009,
           0.07602153943617358,
           6.525182134938232,
           11.872030408615775,
           6.955970858409883,
           2.1792841305036426,
           9.344314222363003,
           2.3439974659486853,
           8.292682926829269,
           7.260057016154578,
           3.199239784605638,
           3.0218561925879,
           1.9512195121951221,
           1.0706366803927778,
           6.455495723788407,
           1.6724738675958188,
           3.9911308203991127,
           4.111498257839721,
           14.665821982895153,
           1.355717453278429,
           14.000633512828635,
           8.970541653468482,
           11.149825783972126,
           0.34209692746278114,
           2.324992081089642,
           7.747861894203357,
           2.920494140006335,
           4.4219195438707635,
           8.127969591384225,
           5.745961355717453,
           0.14570795058599936,
           5.530566993981629,
           4.098828001267025,
           3.0978777320240733,
           8.375039594551788,
           5.435540069686412,
           8.926195755464049,
           9.166930630345265,
           3.8707633829585046,
           2.103262591067469,
           3.699714919227114,
           2.4390243902439024,
           1.3937282229965158,
           0.2787456445993031,
           1.0769718086791258,
           2.6607538802660757,
           8.108964206525183,
           0.2660753880266075,
           2.090592334494773,
           7.963256255939182,
           2.5150459296800762,
           3.300601837187203,
           1.3113715552739944,
           2.103262591067469,
           0.8805828318023441,
           9.572378840671524,
           0.17738359201773835,
           12.853975292999683,
           0.6841938549255623,
           0.8742477035159962,
           0.4181184668989547,
           2.1539436173582516,
           2.8127969591384225,
           3.592017738359202,
           0.19638897687678175,
           1.5140956604371238,
           2.70509977827051,
           9.889135254988913,
           6.436490338929364,
           0.6145074437757365,
           4.396579030725372,
           2.166613873930947,
           1.6407982261640799,
           1.1593284764016472,
           7.038327526132404,
           0.34209692746278114,
           3.218245169464682,
           1.0833069369654735,
           12.853975292999683,
           2.0335761799176435,
           12.328159645232816,
           1.7991764333227749,
           4.0038010769718095,
           2.0842572062084255,
           1.165663604687995,
           0.3737725688945201,
           3.1865695280329427,
           0.7285397529299968,
           3.560342096927463,
           2.800126702565727,
           1.8435223313272093,
           8.070953436807095,
           3.1612290148875513,
           2.7304402914159014,
           7.633829585049098,
           4.409249287298068,
           2.2426354133671205,
           0.7792207792207793,
           1.3810579664238203,
           7.646499841621793,
           3.807412100095027,
           6.170414950902756,
           6.99398162812797,
           2.204624643649034,
           4.649984162179284,
           8.609439341146658,
           3.5413367120684196,
           3.180234399746595,
           3.6110231232182453,
           1.3113715552739944,
           0.8362369337979094,
           1.1846689895470384,
           4.497941083306936,
           1.3050364269876467,
           0.6271777003484321,
           2.363002850807729,
           8.419385492556225,
           3.300601837187203,
           7.139689578713969,
           13.26575863161229,
           9.572378840671524,
           10.624010136205257,
           8.97687678175483,
           5.068102629078239,
           10.04751346214761,
           1.1403230915426037,
           2.6354133671206843,
           0.8995882166613874,
           2.559391827684511,
           1.7801710484637314,
           2.0145707950586003,
           11.213177066835604,
           3.6046879949318975,
           2.305986696230599,
           7.665505226480836,
           3.674374406081723,
           1.9005384859043397,
           6.974976243268927,
           10.300918593601521,
           2.0082356667722525,
           7.025657269559709,
           6.664554957237884,
           6.15774469433006,
           0.13303769401330376,
           3.4209692746278115,
           3.6933797909407664,
           0.15837820715869497
          ]
         },
         "mode": "markers",
         "name": "",
         "showlegend": false,
         "subplot": "mapbox",
         "type": "scattermapbox"
        }
       ],
       "layout": {
        "coloraxis": {
         "colorbar": {
          "title": {
           "text": "Crime Count"
          }
         },
         "colorscale": [
          [
           0,
           "#000000"
          ],
          [
           0.0625,
           "#001f4d"
          ],
          [
           0.125,
           "#003786"
          ],
          [
           0.1875,
           "#0e58a8"
          ],
          [
           0.25,
           "#217eb8"
          ],
          [
           0.3125,
           "#30a4ca"
          ],
          [
           0.375,
           "#54c8df"
          ],
          [
           0.4375,
           "#9be4ef"
          ],
          [
           0.5,
           "#e1e9d1"
          ],
          [
           0.5625,
           "#f3d573"
          ],
          [
           0.625,
           "#e7b000"
          ],
          [
           0.6875,
           "#da8200"
          ],
          [
           0.75,
           "#c65400"
          ],
          [
           0.8125,
           "#ac2301"
          ],
          [
           0.875,
           "#820000"
          ],
          [
           0.9375,
           "#4c0000"
          ],
          [
           1,
           "#000000"
          ]
         ]
        },
        "height": 1000,
        "legend": {
         "tracegroupgap": 0
        },
        "mapbox": {
         "center": {
          "lat": 51.03961343362299,
          "lon": -114.06543729150538
         },
         "domain": {
          "x": [
           0,
           1
          ],
          "y": [
           0,
           1
          ]
         },
         "style": "open-street-map",
         "zoom": 10
        },
        "template": {
         "data": {
          "bar": [
           {
            "error_x": {
             "color": "#2a3f5f"
            },
            "error_y": {
             "color": "#2a3f5f"
            },
            "marker": {
             "line": {
              "color": "#E5ECF6",
              "width": 0.5
             },
             "pattern": {
              "fillmode": "overlay",
              "size": 10,
              "solidity": 0.2
             }
            },
            "type": "bar"
           }
          ],
          "barpolar": [
           {
            "marker": {
             "line": {
              "color": "#E5ECF6",
              "width": 0.5
             },
             "pattern": {
              "fillmode": "overlay",
              "size": 10,
              "solidity": 0.2
             }
            },
            "type": "barpolar"
           }
          ],
          "carpet": [
           {
            "aaxis": {
             "endlinecolor": "#2a3f5f",
             "gridcolor": "white",
             "linecolor": "white",
             "minorgridcolor": "white",
             "startlinecolor": "#2a3f5f"
            },
            "baxis": {
             "endlinecolor": "#2a3f5f",
             "gridcolor": "white",
             "linecolor": "white",
             "minorgridcolor": "white",
             "startlinecolor": "#2a3f5f"
            },
            "type": "carpet"
           }
          ],
          "choropleth": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "type": "choropleth"
           }
          ],
          "contour": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "contour"
           }
          ],
          "contourcarpet": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "type": "contourcarpet"
           }
          ],
          "heatmap": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "heatmap"
           }
          ],
          "heatmapgl": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "heatmapgl"
           }
          ],
          "histogram": [
           {
            "marker": {
             "pattern": {
              "fillmode": "overlay",
              "size": 10,
              "solidity": 0.2
             }
            },
            "type": "histogram"
           }
          ],
          "histogram2d": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "histogram2d"
           }
          ],
          "histogram2dcontour": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "histogram2dcontour"
           }
          ],
          "mesh3d": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "type": "mesh3d"
           }
          ],
          "parcoords": [
           {
            "line": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "parcoords"
           }
          ],
          "pie": [
           {
            "automargin": true,
            "type": "pie"
           }
          ],
          "scatter": [
           {
            "fillpattern": {
             "fillmode": "overlay",
             "size": 10,
             "solidity": 0.2
            },
            "type": "scatter"
           }
          ],
          "scatter3d": [
           {
            "line": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatter3d"
           }
          ],
          "scattercarpet": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattercarpet"
           }
          ],
          "scattergeo": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattergeo"
           }
          ],
          "scattergl": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattergl"
           }
          ],
          "scattermapbox": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattermapbox"
           }
          ],
          "scatterpolar": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatterpolar"
           }
          ],
          "scatterpolargl": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatterpolargl"
           }
          ],
          "scatterternary": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatterternary"
           }
          ],
          "surface": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "surface"
           }
          ],
          "table": [
           {
            "cells": {
             "fill": {
              "color": "#EBF0F8"
             },
             "line": {
              "color": "white"
             }
            },
            "header": {
             "fill": {
              "color": "#C8D4E3"
             },
             "line": {
              "color": "white"
             }
            },
            "type": "table"
           }
          ]
         },
         "layout": {
          "annotationdefaults": {
           "arrowcolor": "#2a3f5f",
           "arrowhead": 0,
           "arrowwidth": 1
          },
          "autotypenumbers": "strict",
          "coloraxis": {
           "colorbar": {
            "outlinewidth": 0,
            "ticks": ""
           }
          },
          "colorscale": {
           "diverging": [
            [
             0,
             "#8e0152"
            ],
            [
             0.1,
             "#c51b7d"
            ],
            [
             0.2,
             "#de77ae"
            ],
            [
             0.3,
             "#f1b6da"
            ],
            [
             0.4,
             "#fde0ef"
            ],
            [
             0.5,
             "#f7f7f7"
            ],
            [
             0.6,
             "#e6f5d0"
            ],
            [
             0.7,
             "#b8e186"
            ],
            [
             0.8,
             "#7fbc41"
            ],
            [
             0.9,
             "#4d9221"
            ],
            [
             1,
             "#276419"
            ]
           ],
           "sequential": [
            [
             0,
             "#0d0887"
            ],
            [
             0.1111111111111111,
             "#46039f"
            ],
            [
             0.2222222222222222,
             "#7201a8"
            ],
            [
             0.3333333333333333,
             "#9c179e"
            ],
            [
             0.4444444444444444,
             "#bd3786"
            ],
            [
             0.5555555555555556,
             "#d8576b"
            ],
            [
             0.6666666666666666,
             "#ed7953"
            ],
            [
             0.7777777777777778,
             "#fb9f3a"
            ],
            [
             0.8888888888888888,
             "#fdca26"
            ],
            [
             1,
             "#f0f921"
            ]
           ],
           "sequentialminus": [
            [
             0,
             "#0d0887"
            ],
            [
             0.1111111111111111,
             "#46039f"
            ],
            [
             0.2222222222222222,
             "#7201a8"
            ],
            [
             0.3333333333333333,
             "#9c179e"
            ],
            [
             0.4444444444444444,
             "#bd3786"
            ],
            [
             0.5555555555555556,
             "#d8576b"
            ],
            [
             0.6666666666666666,
             "#ed7953"
            ],
            [
             0.7777777777777778,
             "#fb9f3a"
            ],
            [
             0.8888888888888888,
             "#fdca26"
            ],
            [
             1,
             "#f0f921"
            ]
           ]
          },
          "colorway": [
           "#636efa",
           "#EF553B",
           "#00cc96",
           "#ab63fa",
           "#FFA15A",
           "#19d3f3",
           "#FF6692",
           "#B6E880",
           "#FF97FF",
           "#FECB52"
          ],
          "font": {
           "color": "#2a3f5f"
          },
          "geo": {
           "bgcolor": "white",
           "lakecolor": "white",
           "landcolor": "#E5ECF6",
           "showlakes": true,
           "showland": true,
           "subunitcolor": "white"
          },
          "hoverlabel": {
           "align": "left"
          },
          "hovermode": "closest",
          "mapbox": {
           "style": "light"
          },
          "paper_bgcolor": "white",
          "plot_bgcolor": "#E5ECF6",
          "polar": {
           "angularaxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           },
           "bgcolor": "#E5ECF6",
           "radialaxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           }
          },
          "scene": {
           "xaxis": {
            "backgroundcolor": "#E5ECF6",
            "gridcolor": "white",
            "gridwidth": 2,
            "linecolor": "white",
            "showbackground": true,
            "ticks": "",
            "zerolinecolor": "white"
           },
           "yaxis": {
            "backgroundcolor": "#E5ECF6",
            "gridcolor": "white",
            "gridwidth": 2,
            "linecolor": "white",
            "showbackground": true,
            "ticks": "",
            "zerolinecolor": "white"
           },
           "zaxis": {
            "backgroundcolor": "#E5ECF6",
            "gridcolor": "white",
            "gridwidth": 2,
            "linecolor": "white",
            "showbackground": true,
            "ticks": "",
            "zerolinecolor": "white"
           }
          },
          "shapedefaults": {
           "line": {
            "color": "#2a3f5f"
           }
          },
          "ternary": {
           "aaxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           },
           "baxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           },
           "bgcolor": "#E5ECF6",
           "caxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           }
          },
          "title": {
           "x": 0.05
          },
          "xaxis": {
           "automargin": true,
           "gridcolor": "white",
           "linecolor": "white",
           "ticks": "",
           "title": {
            "standoff": 15
           },
           "zerolinecolor": "white",
           "zerolinewidth": 2
          },
          "yaxis": {
           "automargin": true,
           "gridcolor": "white",
           "linecolor": "white",
           "ticks": "",
           "title": {
            "standoff": 15
           },
           "zerolinecolor": "white",
           "zerolinewidth": 2
          }
         }
        },
        "title": {
         "text": "Crime Counts by Community Location"
        },
        "width": 2000
       }
      }
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "import plotly.express as px\n",
    "geo_data = df_clean.groupby('Community Name').agg(\n",
    "    {'Crime Count': 'sum', 'Latitude': 'mean', 'Longitude': 'mean'}).reset_index()\n",
    "\n",
    "fig = px.scatter_mapbox(\n",
    "    geo_data,\n",
    "    lat='Latitude',\n",
    "    lon='Longitude',\n",
    "    color='Crime Count',  \n",
    "    hover_name='Community Name',  \n",
    "    color_continuous_scale=px.colors.cyclical.IceFire,  \n",
    "    zoom=10,\n",
    "    mapbox_style=\"open-street-map\",  \n",
    "    title='Crime Counts by Community Location',\n",
    "    width=2000,\n",
    "    height=1000\n",
    ")\n",
    "\n",
    "fig.update_traces(marker=dict(\n",
    "    size=geo_data['Crime Count'] / geo_data['Crime Count'].max() * 60,  \n",
    "    color=geo_data['Crime Count'],  \n",
    "    colorscale='Viridis',  \n",
    "    showscale=True  \n",
    "))\n",
    "\n",
    "fig.show()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "As we can see here 'Centre' is the location with most crime in Calgary, followed by other regions"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Crimes over time (by Sector and Type of Crime)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "To solve this question, I am going to display two line charts to show crimes over time by Sector and by type of crime"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/vnd.plotly.v1+json": {
       "config": {
        "plotlyServerURL": "https://plot.ly"
       },
       "data": [
        {
         "hovertemplate": "Sector=CENTRE<br>Date=%{x}<br>Crime Count=%{y}<extra></extra>",
         "legendgroup": "CENTRE",
         "line": {
          "color": "#636efa",
          "dash": "solid"
         },
         "marker": {
          "symbol": "circle"
         },
         "mode": "lines",
         "name": "CENTRE",
         "orientation": "v",
         "showlegend": true,
         "type": "scatter",
         "x": [
          "2018-01-01T00:00:00",
          "2018-02-01T00:00:00",
          "2018-03-01T00:00:00",
          "2018-04-01T00:00:00",
          "2018-05-01T00:00:00",
          "2018-06-01T00:00:00",
          "2018-07-01T00:00:00",
          "2018-08-01T00:00:00",
          "2018-09-01T00:00:00",
          "2018-10-01T00:00:00",
          "2018-11-01T00:00:00",
          "2018-12-01T00:00:00",
          "2019-01-01T00:00:00",
          "2019-02-01T00:00:00",
          "2019-03-01T00:00:00",
          "2019-04-01T00:00:00",
          "2019-05-01T00:00:00",
          "2019-06-01T00:00:00",
          "2019-07-01T00:00:00",
          "2019-08-01T00:00:00",
          "2019-09-01T00:00:00",
          "2019-10-01T00:00:00",
          "2019-11-01T00:00:00",
          "2019-12-01T00:00:00",
          "2020-01-01T00:00:00",
          "2020-02-01T00:00:00",
          "2020-03-01T00:00:00",
          "2020-04-01T00:00:00",
          "2020-05-01T00:00:00",
          "2020-06-01T00:00:00",
          "2020-07-01T00:00:00",
          "2020-08-01T00:00:00",
          "2020-09-01T00:00:00",
          "2020-10-01T00:00:00",
          "2020-11-01T00:00:00",
          "2020-12-01T00:00:00",
          "2021-01-01T00:00:00",
          "2021-02-01T00:00:00",
          "2021-03-01T00:00:00",
          "2021-04-01T00:00:00",
          "2021-05-01T00:00:00",
          "2021-06-01T00:00:00",
          "2021-07-01T00:00:00",
          "2021-08-01T00:00:00",
          "2021-09-01T00:00:00",
          "2021-10-01T00:00:00",
          "2021-11-01T00:00:00",
          "2021-12-01T00:00:00",
          "2022-01-01T00:00:00",
          "2022-02-01T00:00:00",
          "2022-03-01T00:00:00",
          "2022-04-01T00:00:00",
          "2022-05-01T00:00:00",
          "2022-06-01T00:00:00",
          "2022-07-01T00:00:00",
          "2022-08-01T00:00:00",
          "2022-09-01T00:00:00",
          "2022-10-01T00:00:00",
          "2022-11-01T00:00:00",
          "2022-12-01T00:00:00",
          "2023-01-01T00:00:00",
          "2023-02-01T00:00:00",
          "2023-03-01T00:00:00",
          "2023-04-01T00:00:00",
          "2023-05-01T00:00:00",
          "2023-06-01T00:00:00",
          "2023-07-01T00:00:00",
          "2023-08-01T00:00:00",
          "2023-09-01T00:00:00",
          "2023-10-01T00:00:00",
          "2023-11-01T00:00:00",
          "2023-12-01T00:00:00"
         ],
         "xaxis": "x",
         "y": [
          778,
          635,
          811,
          826,
          976,
          949,
          1185,
          1143,
          1109,
          1032,
          960,
          960,
          1038,
          838,
          945,
          1015,
          1122,
          1158,
          1313,
          1346,
          1423,
          1173,
          1150,
          952,
          1038,
          992,
          950,
          914,
          684,
          798,
          1008,
          1208,
          1133,
          924,
          785,
          630,
          727,
          584,
          727,
          808,
          790,
          778,
          961,
          974,
          926,
          874,
          816,
          737,
          854,
          747,
          909,
          849,
          878,
          924,
          928,
          910,
          818,
          826,
          726,
          674,
          754,
          582,
          711,
          732,
          849,
          847,
          892,
          835,
          612,
          595,
          666,
          531
         ],
         "yaxis": "y"
        },
        {
         "hovertemplate": "Sector=EAST<br>Date=%{x}<br>Crime Count=%{y}<extra></extra>",
         "legendgroup": "EAST",
         "line": {
          "color": "#EF553B",
          "dash": "solid"
         },
         "marker": {
          "symbol": "circle"
         },
         "mode": "lines",
         "name": "EAST",
         "orientation": "v",
         "showlegend": true,
         "type": "scatter",
         "x": [
          "2018-01-01T00:00:00",
          "2018-02-01T00:00:00",
          "2018-03-01T00:00:00",
          "2018-04-01T00:00:00",
          "2018-05-01T00:00:00",
          "2018-06-01T00:00:00",
          "2018-07-01T00:00:00",
          "2018-08-01T00:00:00",
          "2018-09-01T00:00:00",
          "2018-10-01T00:00:00",
          "2018-11-01T00:00:00",
          "2018-12-01T00:00:00",
          "2019-01-01T00:00:00",
          "2019-02-01T00:00:00",
          "2019-03-01T00:00:00",
          "2019-04-01T00:00:00",
          "2019-05-01T00:00:00",
          "2019-06-01T00:00:00",
          "2019-07-01T00:00:00",
          "2019-08-01T00:00:00",
          "2019-09-01T00:00:00",
          "2019-10-01T00:00:00",
          "2019-11-01T00:00:00",
          "2019-12-01T00:00:00",
          "2020-01-01T00:00:00",
          "2020-02-01T00:00:00",
          "2020-03-01T00:00:00",
          "2020-04-01T00:00:00",
          "2020-05-01T00:00:00",
          "2020-06-01T00:00:00",
          "2020-07-01T00:00:00",
          "2020-08-01T00:00:00",
          "2020-09-01T00:00:00",
          "2020-10-01T00:00:00",
          "2020-11-01T00:00:00",
          "2020-12-01T00:00:00",
          "2021-01-01T00:00:00",
          "2021-02-01T00:00:00",
          "2021-03-01T00:00:00",
          "2021-04-01T00:00:00",
          "2021-05-01T00:00:00",
          "2021-06-01T00:00:00",
          "2021-07-01T00:00:00",
          "2021-08-01T00:00:00",
          "2021-09-01T00:00:00",
          "2021-10-01T00:00:00",
          "2021-11-01T00:00:00",
          "2021-12-01T00:00:00",
          "2022-01-01T00:00:00",
          "2022-02-01T00:00:00",
          "2022-03-01T00:00:00",
          "2022-04-01T00:00:00",
          "2022-05-01T00:00:00",
          "2022-06-01T00:00:00",
          "2022-07-01T00:00:00",
          "2022-08-01T00:00:00",
          "2022-09-01T00:00:00",
          "2022-10-01T00:00:00",
          "2022-11-01T00:00:00",
          "2022-12-01T00:00:00",
          "2023-01-01T00:00:00",
          "2023-02-01T00:00:00",
          "2023-03-01T00:00:00",
          "2023-04-01T00:00:00",
          "2023-05-01T00:00:00",
          "2023-06-01T00:00:00",
          "2023-07-01T00:00:00",
          "2023-08-01T00:00:00",
          "2023-09-01T00:00:00",
          "2023-10-01T00:00:00",
          "2023-11-01T00:00:00",
          "2023-12-01T00:00:00"
         ],
         "xaxis": "x",
         "y": [
          190,
          210,
          193,
          212,
          279,
          222,
          267,
          267,
          217,
          262,
          234,
          187,
          177,
          165,
          244,
          232,
          249,
          256,
          326,
          323,
          283,
          232,
          254,
          239,
          211,
          206,
          235,
          206,
          149,
          191,
          247,
          219,
          227,
          214,
          196,
          167,
          203,
          172,
          187,
          202,
          193,
          230,
          247,
          293,
          262,
          204,
          204,
          214,
          263,
          262,
          288,
          246,
          245,
          291,
          236,
          255,
          213,
          218,
          194,
          200,
          167,
          176,
          221,
          180,
          226,
          224,
          190,
          244,
          215,
          171,
          195,
          183
         ],
         "yaxis": "y"
        },
        {
         "hovertemplate": "Sector=NORTH<br>Date=%{x}<br>Crime Count=%{y}<extra></extra>",
         "legendgroup": "NORTH",
         "line": {
          "color": "#00cc96",
          "dash": "solid"
         },
         "marker": {
          "symbol": "circle"
         },
         "mode": "lines",
         "name": "NORTH",
         "orientation": "v",
         "showlegend": true,
         "type": "scatter",
         "x": [
          "2018-01-01T00:00:00",
          "2018-02-01T00:00:00",
          "2018-03-01T00:00:00",
          "2018-04-01T00:00:00",
          "2018-05-01T00:00:00",
          "2018-06-01T00:00:00",
          "2018-07-01T00:00:00",
          "2018-08-01T00:00:00",
          "2018-09-01T00:00:00",
          "2018-10-01T00:00:00",
          "2018-11-01T00:00:00",
          "2018-12-01T00:00:00",
          "2019-01-01T00:00:00",
          "2019-02-01T00:00:00",
          "2019-03-01T00:00:00",
          "2019-04-01T00:00:00",
          "2019-05-01T00:00:00",
          "2019-06-01T00:00:00",
          "2019-07-01T00:00:00",
          "2019-08-01T00:00:00",
          "2019-09-01T00:00:00",
          "2019-10-01T00:00:00",
          "2019-11-01T00:00:00",
          "2019-12-01T00:00:00",
          "2020-01-01T00:00:00",
          "2020-02-01T00:00:00",
          "2020-03-01T00:00:00",
          "2020-04-01T00:00:00",
          "2020-05-01T00:00:00",
          "2020-06-01T00:00:00",
          "2020-07-01T00:00:00",
          "2020-08-01T00:00:00",
          "2020-09-01T00:00:00",
          "2020-10-01T00:00:00",
          "2020-11-01T00:00:00",
          "2020-12-01T00:00:00",
          "2021-01-01T00:00:00",
          "2021-02-01T00:00:00",
          "2021-03-01T00:00:00",
          "2021-04-01T00:00:00",
          "2021-05-01T00:00:00",
          "2021-06-01T00:00:00",
          "2021-07-01T00:00:00",
          "2021-08-01T00:00:00",
          "2021-09-01T00:00:00",
          "2021-10-01T00:00:00",
          "2021-11-01T00:00:00",
          "2021-12-01T00:00:00",
          "2022-01-01T00:00:00",
          "2022-02-01T00:00:00",
          "2022-03-01T00:00:00",
          "2022-04-01T00:00:00",
          "2022-05-01T00:00:00",
          "2022-06-01T00:00:00",
          "2022-07-01T00:00:00",
          "2022-08-01T00:00:00",
          "2022-09-01T00:00:00",
          "2022-10-01T00:00:00",
          "2022-11-01T00:00:00",
          "2022-12-01T00:00:00",
          "2023-01-01T00:00:00",
          "2023-02-01T00:00:00",
          "2023-03-01T00:00:00",
          "2023-04-01T00:00:00",
          "2023-05-01T00:00:00",
          "2023-06-01T00:00:00",
          "2023-07-01T00:00:00",
          "2023-08-01T00:00:00",
          "2023-09-01T00:00:00",
          "2023-10-01T00:00:00",
          "2023-11-01T00:00:00",
          "2023-12-01T00:00:00"
         ],
         "xaxis": "x",
         "y": [
          207,
          154,
          180,
          167,
          234,
          215,
          266,
          286,
          220,
          236,
          178,
          203,
          216,
          164,
          193,
          184,
          221,
          188,
          217,
          256,
          248,
          204,
          201,
          195,
          212,
          207,
          215,
          182,
          152,
          166,
          161,
          150,
          162,
          171,
          140,
          145,
          169,
          117,
          163,
          148,
          164,
          136,
          137,
          130,
          152,
          163,
          157,
          146,
          160,
          190,
          224,
          180,
          191,
          236,
          202,
          210,
          164,
          179,
          166,
          173,
          177,
          147,
          183,
          168,
          134,
          160,
          145,
          149,
          191,
          155,
          139,
          150
         ],
         "yaxis": "y"
        },
        {
         "hovertemplate": "Sector=NORTHEAST<br>Date=%{x}<br>Crime Count=%{y}<extra></extra>",
         "legendgroup": "NORTHEAST",
         "line": {
          "color": "#ab63fa",
          "dash": "solid"
         },
         "marker": {
          "symbol": "circle"
         },
         "mode": "lines",
         "name": "NORTHEAST",
         "orientation": "v",
         "showlegend": true,
         "type": "scatter",
         "x": [
          "2018-01-01T00:00:00",
          "2018-02-01T00:00:00",
          "2018-03-01T00:00:00",
          "2018-04-01T00:00:00",
          "2018-05-01T00:00:00",
          "2018-06-01T00:00:00",
          "2018-07-01T00:00:00",
          "2018-08-01T00:00:00",
          "2018-09-01T00:00:00",
          "2018-10-01T00:00:00",
          "2018-11-01T00:00:00",
          "2018-12-01T00:00:00",
          "2019-01-01T00:00:00",
          "2019-02-01T00:00:00",
          "2019-03-01T00:00:00",
          "2019-04-01T00:00:00",
          "2019-05-01T00:00:00",
          "2019-06-01T00:00:00",
          "2019-07-01T00:00:00",
          "2019-08-01T00:00:00",
          "2019-09-01T00:00:00",
          "2019-10-01T00:00:00",
          "2019-11-01T00:00:00",
          "2019-12-01T00:00:00",
          "2020-01-01T00:00:00",
          "2020-02-01T00:00:00",
          "2020-03-01T00:00:00",
          "2020-04-01T00:00:00",
          "2020-05-01T00:00:00",
          "2020-06-01T00:00:00",
          "2020-07-01T00:00:00",
          "2020-08-01T00:00:00",
          "2020-09-01T00:00:00",
          "2020-10-01T00:00:00",
          "2020-11-01T00:00:00",
          "2020-12-01T00:00:00",
          "2021-01-01T00:00:00",
          "2021-02-01T00:00:00",
          "2021-03-01T00:00:00",
          "2021-04-01T00:00:00",
          "2021-05-01T00:00:00",
          "2021-06-01T00:00:00",
          "2021-07-01T00:00:00",
          "2021-08-01T00:00:00",
          "2021-09-01T00:00:00",
          "2021-10-01T00:00:00",
          "2021-11-01T00:00:00",
          "2021-12-01T00:00:00",
          "2022-01-01T00:00:00",
          "2022-02-01T00:00:00",
          "2022-03-01T00:00:00",
          "2022-04-01T00:00:00",
          "2022-05-01T00:00:00",
          "2022-06-01T00:00:00",
          "2022-07-01T00:00:00",
          "2022-08-01T00:00:00",
          "2022-09-01T00:00:00",
          "2022-10-01T00:00:00",
          "2022-11-01T00:00:00",
          "2022-12-01T00:00:00",
          "2023-01-01T00:00:00",
          "2023-02-01T00:00:00",
          "2023-03-01T00:00:00",
          "2023-04-01T00:00:00",
          "2023-05-01T00:00:00",
          "2023-06-01T00:00:00",
          "2023-07-01T00:00:00",
          "2023-08-01T00:00:00",
          "2023-09-01T00:00:00",
          "2023-10-01T00:00:00",
          "2023-11-01T00:00:00",
          "2023-12-01T00:00:00"
         ],
         "xaxis": "x",
         "y": [
          574,
          471,
          519,
          522,
          560,
          554,
          595,
          582,
          527,
          544,
          533,
          564,
          637,
          458,
          491,
          464,
          546,
          581,
          632,
          551,
          649,
          585,
          490,
          512,
          583,
          622,
          512,
          475,
          382,
          467,
          486,
          501,
          481,
          543,
          432,
          444,
          530,
          450,
          437,
          431,
          409,
          414,
          425,
          464,
          465,
          508,
          512,
          530,
          563,
          524,
          647,
          593,
          598,
          567,
          488,
          604,
          516,
          529,
          475,
          531,
          514,
          477,
          567,
          470,
          519,
          527,
          458,
          424,
          454,
          447,
          414,
          326
         ],
         "yaxis": "y"
        },
        {
         "hovertemplate": "Sector=NORTHWEST<br>Date=%{x}<br>Crime Count=%{y}<extra></extra>",
         "legendgroup": "NORTHWEST",
         "line": {
          "color": "#FFA15A",
          "dash": "solid"
         },
         "marker": {
          "symbol": "circle"
         },
         "mode": "lines",
         "name": "NORTHWEST",
         "orientation": "v",
         "showlegend": true,
         "type": "scatter",
         "x": [
          "2018-01-01T00:00:00",
          "2018-02-01T00:00:00",
          "2018-03-01T00:00:00",
          "2018-04-01T00:00:00",
          "2018-05-01T00:00:00",
          "2018-06-01T00:00:00",
          "2018-07-01T00:00:00",
          "2018-08-01T00:00:00",
          "2018-09-01T00:00:00",
          "2018-10-01T00:00:00",
          "2018-11-01T00:00:00",
          "2018-12-01T00:00:00",
          "2019-01-01T00:00:00",
          "2019-02-01T00:00:00",
          "2019-03-01T00:00:00",
          "2019-04-01T00:00:00",
          "2019-05-01T00:00:00",
          "2019-06-01T00:00:00",
          "2019-07-01T00:00:00",
          "2019-08-01T00:00:00",
          "2019-09-01T00:00:00",
          "2019-10-01T00:00:00",
          "2019-11-01T00:00:00",
          "2019-12-01T00:00:00",
          "2020-01-01T00:00:00",
          "2020-02-01T00:00:00",
          "2020-03-01T00:00:00",
          "2020-04-01T00:00:00",
          "2020-05-01T00:00:00",
          "2020-06-01T00:00:00",
          "2020-07-01T00:00:00",
          "2020-08-01T00:00:00",
          "2020-09-01T00:00:00",
          "2020-10-01T00:00:00",
          "2020-11-01T00:00:00",
          "2020-12-01T00:00:00",
          "2021-01-01T00:00:00",
          "2021-02-01T00:00:00",
          "2021-03-01T00:00:00",
          "2021-04-01T00:00:00",
          "2021-05-01T00:00:00",
          "2021-06-01T00:00:00",
          "2021-07-01T00:00:00",
          "2021-08-01T00:00:00",
          "2021-09-01T00:00:00",
          "2021-10-01T00:00:00",
          "2021-11-01T00:00:00",
          "2021-12-01T00:00:00",
          "2022-01-01T00:00:00",
          "2022-02-01T00:00:00",
          "2022-03-01T00:00:00",
          "2022-04-01T00:00:00",
          "2022-05-01T00:00:00",
          "2022-06-01T00:00:00",
          "2022-07-01T00:00:00",
          "2022-08-01T00:00:00",
          "2022-09-01T00:00:00",
          "2022-10-01T00:00:00",
          "2022-11-01T00:00:00",
          "2022-12-01T00:00:00",
          "2023-01-01T00:00:00",
          "2023-02-01T00:00:00",
          "2023-03-01T00:00:00",
          "2023-04-01T00:00:00",
          "2023-05-01T00:00:00",
          "2023-06-01T00:00:00",
          "2023-07-01T00:00:00",
          "2023-08-01T00:00:00",
          "2023-09-01T00:00:00",
          "2023-10-01T00:00:00",
          "2023-11-01T00:00:00",
          "2023-12-01T00:00:00"
         ],
         "xaxis": "x",
         "y": [
          309,
          233,
          266,
          274,
          276,
          289,
          305,
          396,
          309,
          341,
          263,
          187,
          265,
          169,
          215,
          193,
          261,
          279,
          325,
          314,
          292,
          281,
          249,
          252,
          195,
          209,
          206,
          216,
          178,
          212,
          237,
          267,
          220,
          254,
          193,
          177,
          218,
          139,
          171,
          175,
          204,
          207,
          234,
          215,
          209,
          197,
          237,
          236,
          345,
          250,
          355,
          291,
          299,
          295,
          281,
          281,
          266,
          265,
          219,
          242,
          200,
          200,
          255,
          193,
          194,
          214,
          168,
          144,
          185,
          148,
          148,
          156
         ],
         "yaxis": "y"
        },
        {
         "hovertemplate": "Sector=SOUTH<br>Date=%{x}<br>Crime Count=%{y}<extra></extra>",
         "legendgroup": "SOUTH",
         "line": {
          "color": "#19d3f3",
          "dash": "solid"
         },
         "marker": {
          "symbol": "circle"
         },
         "mode": "lines",
         "name": "SOUTH",
         "orientation": "v",
         "showlegend": true,
         "type": "scatter",
         "x": [
          "2018-01-01T00:00:00",
          "2018-02-01T00:00:00",
          "2018-03-01T00:00:00",
          "2018-04-01T00:00:00",
          "2018-05-01T00:00:00",
          "2018-06-01T00:00:00",
          "2018-07-01T00:00:00",
          "2018-08-01T00:00:00",
          "2018-09-01T00:00:00",
          "2018-10-01T00:00:00",
          "2018-11-01T00:00:00",
          "2018-12-01T00:00:00",
          "2019-01-01T00:00:00",
          "2019-02-01T00:00:00",
          "2019-03-01T00:00:00",
          "2019-04-01T00:00:00",
          "2019-05-01T00:00:00",
          "2019-06-01T00:00:00",
          "2019-07-01T00:00:00",
          "2019-08-01T00:00:00",
          "2019-09-01T00:00:00",
          "2019-10-01T00:00:00",
          "2019-11-01T00:00:00",
          "2019-12-01T00:00:00",
          "2020-01-01T00:00:00",
          "2020-02-01T00:00:00",
          "2020-03-01T00:00:00",
          "2020-04-01T00:00:00",
          "2020-05-01T00:00:00",
          "2020-06-01T00:00:00",
          "2020-07-01T00:00:00",
          "2020-08-01T00:00:00",
          "2020-09-01T00:00:00",
          "2020-10-01T00:00:00",
          "2020-11-01T00:00:00",
          "2020-12-01T00:00:00",
          "2021-01-01T00:00:00",
          "2021-02-01T00:00:00",
          "2021-03-01T00:00:00",
          "2021-04-01T00:00:00",
          "2021-05-01T00:00:00",
          "2021-06-01T00:00:00",
          "2021-07-01T00:00:00",
          "2021-08-01T00:00:00",
          "2021-09-01T00:00:00",
          "2021-10-01T00:00:00",
          "2021-11-01T00:00:00",
          "2021-12-01T00:00:00",
          "2022-01-01T00:00:00",
          "2022-02-01T00:00:00",
          "2022-03-01T00:00:00",
          "2022-04-01T00:00:00",
          "2022-05-01T00:00:00",
          "2022-06-01T00:00:00",
          "2022-07-01T00:00:00",
          "2022-08-01T00:00:00",
          "2022-09-01T00:00:00",
          "2022-10-01T00:00:00",
          "2022-11-01T00:00:00",
          "2022-12-01T00:00:00",
          "2023-01-01T00:00:00",
          "2023-02-01T00:00:00",
          "2023-03-01T00:00:00",
          "2023-04-01T00:00:00",
          "2023-05-01T00:00:00",
          "2023-06-01T00:00:00",
          "2023-07-01T00:00:00",
          "2023-08-01T00:00:00",
          "2023-09-01T00:00:00",
          "2023-10-01T00:00:00",
          "2023-11-01T00:00:00",
          "2023-12-01T00:00:00"
         ],
         "xaxis": "x",
         "y": [
          371,
          286,
          295,
          339,
          354,
          336,
          424,
          422,
          437,
          425,
          342,
          352,
          376,
          290,
          342,
          398,
          375,
          488,
          486,
          529,
          484,
          454,
          395,
          338,
          338,
          382,
          344,
          295,
          207,
          221,
          297,
          352,
          394,
          398,
          299,
          254,
          234,
          188,
          245,
          194,
          234,
          217,
          293,
          298,
          253,
          258,
          353,
          249,
          325,
          283,
          367,
          325,
          386,
          369,
          337,
          361,
          311,
          292,
          245,
          260,
          320,
          247,
          353,
          307,
          278,
          303,
          305,
          321,
          300,
          258,
          246,
          204
         ],
         "yaxis": "y"
        },
        {
         "hovertemplate": "Sector=SOUTHEAST<br>Date=%{x}<br>Crime Count=%{y}<extra></extra>",
         "legendgroup": "SOUTHEAST",
         "line": {
          "color": "#FF6692",
          "dash": "solid"
         },
         "marker": {
          "symbol": "circle"
         },
         "mode": "lines",
         "name": "SOUTHEAST",
         "orientation": "v",
         "showlegend": true,
         "type": "scatter",
         "x": [
          "2018-01-01T00:00:00",
          "2018-02-01T00:00:00",
          "2018-03-01T00:00:00",
          "2018-04-01T00:00:00",
          "2018-05-01T00:00:00",
          "2018-06-01T00:00:00",
          "2018-07-01T00:00:00",
          "2018-08-01T00:00:00",
          "2018-09-01T00:00:00",
          "2018-10-01T00:00:00",
          "2018-11-01T00:00:00",
          "2018-12-01T00:00:00",
          "2019-01-01T00:00:00",
          "2019-02-01T00:00:00",
          "2019-03-01T00:00:00",
          "2019-04-01T00:00:00",
          "2019-05-01T00:00:00",
          "2019-06-01T00:00:00",
          "2019-07-01T00:00:00",
          "2019-08-01T00:00:00",
          "2019-09-01T00:00:00",
          "2019-10-01T00:00:00",
          "2019-11-01T00:00:00",
          "2019-12-01T00:00:00",
          "2020-01-01T00:00:00",
          "2020-02-01T00:00:00",
          "2020-03-01T00:00:00",
          "2020-04-01T00:00:00",
          "2020-05-01T00:00:00",
          "2020-06-01T00:00:00",
          "2020-07-01T00:00:00",
          "2020-08-01T00:00:00",
          "2020-09-01T00:00:00",
          "2020-10-01T00:00:00",
          "2020-11-01T00:00:00",
          "2020-12-01T00:00:00",
          "2021-01-01T00:00:00",
          "2021-02-01T00:00:00",
          "2021-03-01T00:00:00",
          "2021-04-01T00:00:00",
          "2021-05-01T00:00:00",
          "2021-06-01T00:00:00",
          "2021-07-01T00:00:00",
          "2021-08-01T00:00:00",
          "2021-09-01T00:00:00",
          "2021-10-01T00:00:00",
          "2021-11-01T00:00:00",
          "2021-12-01T00:00:00",
          "2022-01-01T00:00:00",
          "2022-02-01T00:00:00",
          "2022-03-01T00:00:00",
          "2022-04-01T00:00:00",
          "2022-05-01T00:00:00",
          "2022-06-01T00:00:00",
          "2022-07-01T00:00:00",
          "2022-08-01T00:00:00",
          "2022-09-01T00:00:00",
          "2022-10-01T00:00:00",
          "2022-11-01T00:00:00",
          "2022-12-01T00:00:00",
          "2023-01-01T00:00:00",
          "2023-02-01T00:00:00",
          "2023-03-01T00:00:00",
          "2023-04-01T00:00:00",
          "2023-05-01T00:00:00",
          "2023-06-01T00:00:00",
          "2023-07-01T00:00:00",
          "2023-08-01T00:00:00",
          "2023-09-01T00:00:00",
          "2023-10-01T00:00:00",
          "2023-11-01T00:00:00",
          "2023-12-01T00:00:00"
         ],
         "xaxis": "x",
         "y": [
          173,
          125,
          172,
          156,
          191,
          187,
          213,
          236,
          165,
          158,
          158,
          189,
          195,
          121,
          144,
          168,
          177,
          202,
          212,
          230,
          244,
          211,
          194,
          184,
          254,
          204,
          204,
          167,
          106,
          123,
          153,
          232,
          190,
          168,
          103,
          120,
          116,
          80,
          105,
          120,
          110,
          105,
          124,
          154,
          166,
          163,
          161,
          130,
          128,
          167,
          211,
          136,
          191,
          194,
          169,
          172,
          139,
          170,
          109,
          143,
          137,
          126,
          135,
          142,
          175,
          171,
          148,
          136,
          137,
          126,
          157,
          116
         ],
         "yaxis": "y"
        },
        {
         "hovertemplate": "Sector=WEST<br>Date=%{x}<br>Crime Count=%{y}<extra></extra>",
         "legendgroup": "WEST",
         "line": {
          "color": "#B6E880",
          "dash": "solid"
         },
         "marker": {
          "symbol": "circle"
         },
         "mode": "lines",
         "name": "WEST",
         "orientation": "v",
         "showlegend": true,
         "type": "scatter",
         "x": [
          "2018-01-01T00:00:00",
          "2018-02-01T00:00:00",
          "2018-03-01T00:00:00",
          "2018-04-01T00:00:00",
          "2018-05-01T00:00:00",
          "2018-06-01T00:00:00",
          "2018-07-01T00:00:00",
          "2018-08-01T00:00:00",
          "2018-09-01T00:00:00",
          "2018-10-01T00:00:00",
          "2018-11-01T00:00:00",
          "2018-12-01T00:00:00",
          "2019-01-01T00:00:00",
          "2019-02-01T00:00:00",
          "2019-03-01T00:00:00",
          "2019-04-01T00:00:00",
          "2019-05-01T00:00:00",
          "2019-06-01T00:00:00",
          "2019-07-01T00:00:00",
          "2019-08-01T00:00:00",
          "2019-09-01T00:00:00",
          "2019-10-01T00:00:00",
          "2019-11-01T00:00:00",
          "2019-12-01T00:00:00",
          "2020-01-01T00:00:00",
          "2020-02-01T00:00:00",
          "2020-03-01T00:00:00",
          "2020-04-01T00:00:00",
          "2020-05-01T00:00:00",
          "2020-06-01T00:00:00",
          "2020-07-01T00:00:00",
          "2020-08-01T00:00:00",
          "2020-09-01T00:00:00",
          "2020-10-01T00:00:00",
          "2020-11-01T00:00:00",
          "2020-12-01T00:00:00",
          "2021-01-01T00:00:00",
          "2021-02-01T00:00:00",
          "2021-03-01T00:00:00",
          "2021-04-01T00:00:00",
          "2021-05-01T00:00:00",
          "2021-06-01T00:00:00",
          "2021-07-01T00:00:00",
          "2021-08-01T00:00:00",
          "2021-09-01T00:00:00",
          "2021-10-01T00:00:00",
          "2021-11-01T00:00:00",
          "2021-12-01T00:00:00",
          "2022-01-01T00:00:00",
          "2022-02-01T00:00:00",
          "2022-03-01T00:00:00",
          "2022-04-01T00:00:00",
          "2022-05-01T00:00:00",
          "2022-06-01T00:00:00",
          "2022-07-01T00:00:00",
          "2022-08-01T00:00:00",
          "2022-09-01T00:00:00",
          "2022-10-01T00:00:00",
          "2022-11-01T00:00:00",
          "2022-12-01T00:00:00",
          "2023-01-01T00:00:00",
          "2023-02-01T00:00:00",
          "2023-03-01T00:00:00",
          "2023-04-01T00:00:00",
          "2023-05-01T00:00:00",
          "2023-06-01T00:00:00",
          "2023-07-01T00:00:00",
          "2023-08-01T00:00:00",
          "2023-09-01T00:00:00",
          "2023-10-01T00:00:00",
          "2023-11-01T00:00:00",
          "2023-12-01T00:00:00"
         ],
         "xaxis": "x",
         "y": [
          208,
          153,
          134,
          144,
          189,
          183,
          196,
          225,
          225,
          213,
          138,
          175,
          178,
          115,
          128,
          169,
          214,
          204,
          246,
          243,
          262,
          210,
          167,
          150,
          149,
          192,
          187,
          128,
          122,
          121,
          149,
          163,
          212,
          189,
          163,
          107,
          132,
          115,
          133,
          125,
          117,
          148,
          154,
          143,
          164,
          180,
          150,
          153,
          181,
          163,
          208,
          115,
          132,
          132,
          134,
          138,
          125,
          146,
          77,
          126,
          133,
          103,
          121,
          102,
          147,
          128,
          140,
          121,
          121,
          103,
          92,
          118
         ],
         "yaxis": "y"
        }
       ],
       "layout": {
        "height": 1000,
        "legend": {
         "title": {
          "text": "Sector"
         },
         "tracegroupgap": 0
        },
        "template": {
         "data": {
          "bar": [
           {
            "error_x": {
             "color": "#2a3f5f"
            },
            "error_y": {
             "color": "#2a3f5f"
            },
            "marker": {
             "line": {
              "color": "#E5ECF6",
              "width": 0.5
             },
             "pattern": {
              "fillmode": "overlay",
              "size": 10,
              "solidity": 0.2
             }
            },
            "type": "bar"
           }
          ],
          "barpolar": [
           {
            "marker": {
             "line": {
              "color": "#E5ECF6",
              "width": 0.5
             },
             "pattern": {
              "fillmode": "overlay",
              "size": 10,
              "solidity": 0.2
             }
            },
            "type": "barpolar"
           }
          ],
          "carpet": [
           {
            "aaxis": {
             "endlinecolor": "#2a3f5f",
             "gridcolor": "white",
             "linecolor": "white",
             "minorgridcolor": "white",
             "startlinecolor": "#2a3f5f"
            },
            "baxis": {
             "endlinecolor": "#2a3f5f",
             "gridcolor": "white",
             "linecolor": "white",
             "minorgridcolor": "white",
             "startlinecolor": "#2a3f5f"
            },
            "type": "carpet"
           }
          ],
          "choropleth": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "type": "choropleth"
           }
          ],
          "contour": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "contour"
           }
          ],
          "contourcarpet": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "type": "contourcarpet"
           }
          ],
          "heatmap": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "heatmap"
           }
          ],
          "heatmapgl": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "heatmapgl"
           }
          ],
          "histogram": [
           {
            "marker": {
             "pattern": {
              "fillmode": "overlay",
              "size": 10,
              "solidity": 0.2
             }
            },
            "type": "histogram"
           }
          ],
          "histogram2d": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "histogram2d"
           }
          ],
          "histogram2dcontour": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "histogram2dcontour"
           }
          ],
          "mesh3d": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "type": "mesh3d"
           }
          ],
          "parcoords": [
           {
            "line": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "parcoords"
           }
          ],
          "pie": [
           {
            "automargin": true,
            "type": "pie"
           }
          ],
          "scatter": [
           {
            "fillpattern": {
             "fillmode": "overlay",
             "size": 10,
             "solidity": 0.2
            },
            "type": "scatter"
           }
          ],
          "scatter3d": [
           {
            "line": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatter3d"
           }
          ],
          "scattercarpet": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattercarpet"
           }
          ],
          "scattergeo": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattergeo"
           }
          ],
          "scattergl": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattergl"
           }
          ],
          "scattermapbox": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattermapbox"
           }
          ],
          "scatterpolar": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatterpolar"
           }
          ],
          "scatterpolargl": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatterpolargl"
           }
          ],
          "scatterternary": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatterternary"
           }
          ],
          "surface": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "surface"
           }
          ],
          "table": [
           {
            "cells": {
             "fill": {
              "color": "#EBF0F8"
             },
             "line": {
              "color": "white"
             }
            },
            "header": {
             "fill": {
              "color": "#C8D4E3"
             },
             "line": {
              "color": "white"
             }
            },
            "type": "table"
           }
          ]
         },
         "layout": {
          "annotationdefaults": {
           "arrowcolor": "#2a3f5f",
           "arrowhead": 0,
           "arrowwidth": 1
          },
          "autotypenumbers": "strict",
          "coloraxis": {
           "colorbar": {
            "outlinewidth": 0,
            "ticks": ""
           }
          },
          "colorscale": {
           "diverging": [
            [
             0,
             "#8e0152"
            ],
            [
             0.1,
             "#c51b7d"
            ],
            [
             0.2,
             "#de77ae"
            ],
            [
             0.3,
             "#f1b6da"
            ],
            [
             0.4,
             "#fde0ef"
            ],
            [
             0.5,
             "#f7f7f7"
            ],
            [
             0.6,
             "#e6f5d0"
            ],
            [
             0.7,
             "#b8e186"
            ],
            [
             0.8,
             "#7fbc41"
            ],
            [
             0.9,
             "#4d9221"
            ],
            [
             1,
             "#276419"
            ]
           ],
           "sequential": [
            [
             0,
             "#0d0887"
            ],
            [
             0.1111111111111111,
             "#46039f"
            ],
            [
             0.2222222222222222,
             "#7201a8"
            ],
            [
             0.3333333333333333,
             "#9c179e"
            ],
            [
             0.4444444444444444,
             "#bd3786"
            ],
            [
             0.5555555555555556,
             "#d8576b"
            ],
            [
             0.6666666666666666,
             "#ed7953"
            ],
            [
             0.7777777777777778,
             "#fb9f3a"
            ],
            [
             0.8888888888888888,
             "#fdca26"
            ],
            [
             1,
             "#f0f921"
            ]
           ],
           "sequentialminus": [
            [
             0,
             "#0d0887"
            ],
            [
             0.1111111111111111,
             "#46039f"
            ],
            [
             0.2222222222222222,
             "#7201a8"
            ],
            [
             0.3333333333333333,
             "#9c179e"
            ],
            [
             0.4444444444444444,
             "#bd3786"
            ],
            [
             0.5555555555555556,
             "#d8576b"
            ],
            [
             0.6666666666666666,
             "#ed7953"
            ],
            [
             0.7777777777777778,
             "#fb9f3a"
            ],
            [
             0.8888888888888888,
             "#fdca26"
            ],
            [
             1,
             "#f0f921"
            ]
           ]
          },
          "colorway": [
           "#636efa",
           "#EF553B",
           "#00cc96",
           "#ab63fa",
           "#FFA15A",
           "#19d3f3",
           "#FF6692",
           "#B6E880",
           "#FF97FF",
           "#FECB52"
          ],
          "font": {
           "color": "#2a3f5f"
          },
          "geo": {
           "bgcolor": "white",
           "lakecolor": "white",
           "landcolor": "#E5ECF6",
           "showlakes": true,
           "showland": true,
           "subunitcolor": "white"
          },
          "hoverlabel": {
           "align": "left"
          },
          "hovermode": "closest",
          "mapbox": {
           "style": "light"
          },
          "paper_bgcolor": "white",
          "plot_bgcolor": "#E5ECF6",
          "polar": {
           "angularaxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           },
           "bgcolor": "#E5ECF6",
           "radialaxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           }
          },
          "scene": {
           "xaxis": {
            "backgroundcolor": "#E5ECF6",
            "gridcolor": "white",
            "gridwidth": 2,
            "linecolor": "white",
            "showbackground": true,
            "ticks": "",
            "zerolinecolor": "white"
           },
           "yaxis": {
            "backgroundcolor": "#E5ECF6",
            "gridcolor": "white",
            "gridwidth": 2,
            "linecolor": "white",
            "showbackground": true,
            "ticks": "",
            "zerolinecolor": "white"
           },
           "zaxis": {
            "backgroundcolor": "#E5ECF6",
            "gridcolor": "white",
            "gridwidth": 2,
            "linecolor": "white",
            "showbackground": true,
            "ticks": "",
            "zerolinecolor": "white"
           }
          },
          "shapedefaults": {
           "line": {
            "color": "#2a3f5f"
           }
          },
          "ternary": {
           "aaxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           },
           "baxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           },
           "bgcolor": "#E5ECF6",
           "caxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           }
          },
          "title": {
           "x": 0.05
          },
          "xaxis": {
           "automargin": true,
           "gridcolor": "white",
           "linecolor": "white",
           "ticks": "",
           "title": {
            "standoff": 15
           },
           "zerolinecolor": "white",
           "zerolinewidth": 2
          },
          "yaxis": {
           "automargin": true,
           "gridcolor": "white",
           "linecolor": "white",
           "ticks": "",
           "title": {
            "standoff": 15
           },
           "zerolinecolor": "white",
           "zerolinewidth": 2
          }
         }
        },
        "title": {
         "text": "Crime Count Over Time by Sector"
        },
        "width": 2000,
        "xaxis": {
         "anchor": "y",
         "domain": [
          0,
          1
         ],
         "title": {
          "text": "Date"
         }
        },
        "yaxis": {
         "anchor": "x",
         "domain": [
          0,
          1
         ],
         "title": {
          "text": "Crime count"
         }
        }
       }
      }
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "import pandas as pd\n",
    "import plotly.express as px\n",
    "\n",
    "df_grouped = df_clean.groupby(['Sector', 'Date'])['Crime Count'].sum().reset_index()\n",
    "fig = px.line(df_grouped, x='Date', y='Crime Count', color='Sector', title='Crime Count Over Time by Sector')\n",
    "fig.update_layout(xaxis_title='Date', yaxis_title='Crime count', legend_title='Sector', width=2000, height=1000)\n",
    "fig.show()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "As we can see, crimes in general have been decreasing over time, in addition to highlighting that 'Centre' and 'Northeast' are the areas with the highest amount of crimes historically."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/vnd.plotly.v1+json": {
       "config": {
        "plotlyServerURL": "https://plot.ly"
       },
       "data": [
        {
         "hovertemplate": "Category=Assault (Non-domestic)<br>Date=%{x}<br>Crime Count=%{y}<extra></extra>",
         "legendgroup": "Assault (Non-domestic)",
         "line": {
          "color": "#636efa",
          "dash": "solid"
         },
         "marker": {
          "symbol": "circle"
         },
         "mode": "lines",
         "name": "Assault (Non-domestic)",
         "orientation": "v",
         "showlegend": true,
         "type": "scatter",
         "x": [
          "2018-01-01T00:00:00",
          "2018-02-01T00:00:00",
          "2018-03-01T00:00:00",
          "2018-04-01T00:00:00",
          "2018-05-01T00:00:00",
          "2018-06-01T00:00:00",
          "2018-07-01T00:00:00",
          "2018-08-01T00:00:00",
          "2018-09-01T00:00:00",
          "2018-10-01T00:00:00",
          "2018-11-01T00:00:00",
          "2018-12-01T00:00:00",
          "2019-01-01T00:00:00",
          "2019-02-01T00:00:00",
          "2019-03-01T00:00:00",
          "2019-04-01T00:00:00",
          "2019-05-01T00:00:00",
          "2019-06-01T00:00:00",
          "2019-07-01T00:00:00",
          "2019-08-01T00:00:00",
          "2019-09-01T00:00:00",
          "2019-10-01T00:00:00",
          "2019-11-01T00:00:00",
          "2019-12-01T00:00:00",
          "2020-01-01T00:00:00",
          "2020-02-01T00:00:00",
          "2020-03-01T00:00:00",
          "2020-04-01T00:00:00",
          "2020-05-01T00:00:00",
          "2020-06-01T00:00:00",
          "2020-07-01T00:00:00",
          "2020-08-01T00:00:00",
          "2020-09-01T00:00:00",
          "2020-10-01T00:00:00",
          "2020-11-01T00:00:00",
          "2020-12-01T00:00:00",
          "2021-01-01T00:00:00",
          "2021-02-01T00:00:00",
          "2021-03-01T00:00:00",
          "2021-04-01T00:00:00",
          "2021-05-01T00:00:00",
          "2021-06-01T00:00:00",
          "2021-07-01T00:00:00",
          "2021-08-01T00:00:00",
          "2021-09-01T00:00:00",
          "2021-10-01T00:00:00",
          "2021-11-01T00:00:00",
          "2021-12-01T00:00:00",
          "2022-01-01T00:00:00",
          "2022-02-01T00:00:00",
          "2022-03-01T00:00:00",
          "2022-04-01T00:00:00",
          "2022-05-01T00:00:00",
          "2022-06-01T00:00:00",
          "2022-07-01T00:00:00",
          "2022-08-01T00:00:00",
          "2022-09-01T00:00:00",
          "2022-10-01T00:00:00",
          "2022-11-01T00:00:00",
          "2022-12-01T00:00:00",
          "2023-01-01T00:00:00",
          "2023-02-01T00:00:00",
          "2023-03-01T00:00:00",
          "2023-04-01T00:00:00",
          "2023-05-01T00:00:00",
          "2023-06-01T00:00:00",
          "2023-07-01T00:00:00",
          "2023-08-01T00:00:00",
          "2023-09-01T00:00:00",
          "2023-10-01T00:00:00",
          "2023-11-01T00:00:00",
          "2023-12-01T00:00:00"
         ],
         "xaxis": "x",
         "y": [
          256,
          260,
          291,
          294,
          402,
          378,
          329,
          351,
          296,
          337,
          311,
          323,
          345,
          251,
          319,
          304,
          330,
          350,
          379,
          332,
          330,
          361,
          313,
          299,
          278,
          330,
          247,
          222,
          229,
          335,
          392,
          373,
          351,
          289,
          247,
          236,
          277,
          243,
          290,
          291,
          312,
          389,
          384,
          397,
          391,
          371,
          392,
          299,
          321,
          309,
          365,
          313,
          366,
          391,
          377,
          380,
          350,
          370,
          299,
          298,
          299,
          323,
          383,
          387,
          431,
          451,
          418,
          433,
          393,
          355,
          357,
          346
         ],
         "yaxis": "y"
        },
        {
         "hovertemplate": "Category=Break & Enter - Commercial<br>Date=%{x}<br>Crime Count=%{y}<extra></extra>",
         "legendgroup": "Break & Enter - Commercial",
         "line": {
          "color": "#EF553B",
          "dash": "solid"
         },
         "marker": {
          "symbol": "circle"
         },
         "mode": "lines",
         "name": "Break & Enter - Commercial",
         "orientation": "v",
         "showlegend": true,
         "type": "scatter",
         "x": [
          "2018-01-01T00:00:00",
          "2018-02-01T00:00:00",
          "2018-03-01T00:00:00",
          "2018-04-01T00:00:00",
          "2018-05-01T00:00:00",
          "2018-06-01T00:00:00",
          "2018-07-01T00:00:00",
          "2018-08-01T00:00:00",
          "2018-09-01T00:00:00",
          "2018-10-01T00:00:00",
          "2018-11-01T00:00:00",
          "2018-12-01T00:00:00",
          "2019-01-01T00:00:00",
          "2019-02-01T00:00:00",
          "2019-03-01T00:00:00",
          "2019-04-01T00:00:00",
          "2019-05-01T00:00:00",
          "2019-06-01T00:00:00",
          "2019-07-01T00:00:00",
          "2019-08-01T00:00:00",
          "2019-09-01T00:00:00",
          "2019-10-01T00:00:00",
          "2019-11-01T00:00:00",
          "2019-12-01T00:00:00",
          "2020-01-01T00:00:00",
          "2020-02-01T00:00:00",
          "2020-03-01T00:00:00",
          "2020-04-01T00:00:00",
          "2020-05-01T00:00:00",
          "2020-06-01T00:00:00",
          "2020-07-01T00:00:00",
          "2020-08-01T00:00:00",
          "2020-09-01T00:00:00",
          "2020-10-01T00:00:00",
          "2020-11-01T00:00:00",
          "2020-12-01T00:00:00",
          "2021-01-01T00:00:00",
          "2021-02-01T00:00:00",
          "2021-03-01T00:00:00",
          "2021-04-01T00:00:00",
          "2021-05-01T00:00:00",
          "2021-06-01T00:00:00",
          "2021-07-01T00:00:00",
          "2021-08-01T00:00:00",
          "2021-09-01T00:00:00",
          "2021-10-01T00:00:00",
          "2021-11-01T00:00:00",
          "2021-12-01T00:00:00",
          "2022-01-01T00:00:00",
          "2022-02-01T00:00:00",
          "2022-03-01T00:00:00",
          "2022-04-01T00:00:00",
          "2022-05-01T00:00:00",
          "2022-06-01T00:00:00",
          "2022-07-01T00:00:00",
          "2022-08-01T00:00:00",
          "2022-09-01T00:00:00",
          "2022-10-01T00:00:00",
          "2022-11-01T00:00:00",
          "2022-12-01T00:00:00",
          "2023-01-01T00:00:00",
          "2023-02-01T00:00:00",
          "2023-03-01T00:00:00",
          "2023-04-01T00:00:00",
          "2023-05-01T00:00:00",
          "2023-06-01T00:00:00",
          "2023-07-01T00:00:00",
          "2023-08-01T00:00:00",
          "2023-09-01T00:00:00",
          "2023-10-01T00:00:00",
          "2023-11-01T00:00:00",
          "2023-12-01T00:00:00"
         ],
         "xaxis": "x",
         "y": [
          393,
          313,
          411,
          384,
          450,
          363,
          491,
          545,
          440,
          411,
          390,
          463,
          539,
          416,
          466,
          423,
          474,
          492,
          462,
          554,
          639,
          517,
          423,
          377,
          393,
          442,
          508,
          399,
          263,
          290,
          313,
          368,
          316,
          327,
          278,
          306,
          293,
          235,
          256,
          269,
          261,
          189,
          284,
          285,
          301,
          285,
          257,
          297,
          410,
          403,
          487,
          379,
          362,
          381,
          299,
          356,
          311,
          299,
          265,
          321,
          343,
          244,
          364,
          272,
          263,
          266,
          258,
          280,
          259,
          273,
          305,
          217
         ],
         "yaxis": "y"
        },
        {
         "hovertemplate": "Category=Break & Enter - Dwelling<br>Date=%{x}<br>Crime Count=%{y}<extra></extra>",
         "legendgroup": "Break & Enter - Dwelling",
         "line": {
          "color": "#00cc96",
          "dash": "solid"
         },
         "marker": {
          "symbol": "circle"
         },
         "mode": "lines",
         "name": "Break & Enter - Dwelling",
         "orientation": "v",
         "showlegend": true,
         "type": "scatter",
         "x": [
          "2018-01-01T00:00:00",
          "2018-02-01T00:00:00",
          "2018-03-01T00:00:00",
          "2018-04-01T00:00:00",
          "2018-05-01T00:00:00",
          "2018-06-01T00:00:00",
          "2018-07-01T00:00:00",
          "2018-08-01T00:00:00",
          "2018-09-01T00:00:00",
          "2018-10-01T00:00:00",
          "2018-11-01T00:00:00",
          "2018-12-01T00:00:00",
          "2019-01-01T00:00:00",
          "2019-02-01T00:00:00",
          "2019-03-01T00:00:00",
          "2019-04-01T00:00:00",
          "2019-05-01T00:00:00",
          "2019-06-01T00:00:00",
          "2019-07-01T00:00:00",
          "2019-08-01T00:00:00",
          "2019-09-01T00:00:00",
          "2019-10-01T00:00:00",
          "2019-11-01T00:00:00",
          "2019-12-01T00:00:00",
          "2020-01-01T00:00:00",
          "2020-02-01T00:00:00",
          "2020-03-01T00:00:00",
          "2020-04-01T00:00:00",
          "2020-05-01T00:00:00",
          "2020-06-01T00:00:00",
          "2020-07-01T00:00:00",
          "2020-08-01T00:00:00",
          "2020-09-01T00:00:00",
          "2020-10-01T00:00:00",
          "2020-11-01T00:00:00",
          "2020-12-01T00:00:00",
          "2021-01-01T00:00:00",
          "2021-02-01T00:00:00",
          "2021-03-01T00:00:00",
          "2021-04-01T00:00:00",
          "2021-05-01T00:00:00",
          "2021-06-01T00:00:00",
          "2021-07-01T00:00:00",
          "2021-08-01T00:00:00",
          "2021-09-01T00:00:00",
          "2021-10-01T00:00:00",
          "2021-11-01T00:00:00",
          "2021-12-01T00:00:00",
          "2022-01-01T00:00:00",
          "2022-02-01T00:00:00",
          "2022-03-01T00:00:00",
          "2022-04-01T00:00:00",
          "2022-05-01T00:00:00",
          "2022-06-01T00:00:00",
          "2022-07-01T00:00:00",
          "2022-08-01T00:00:00",
          "2022-09-01T00:00:00",
          "2022-10-01T00:00:00",
          "2022-11-01T00:00:00",
          "2022-12-01T00:00:00",
          "2023-01-01T00:00:00",
          "2023-02-01T00:00:00",
          "2023-03-01T00:00:00",
          "2023-04-01T00:00:00",
          "2023-05-01T00:00:00",
          "2023-06-01T00:00:00",
          "2023-07-01T00:00:00",
          "2023-08-01T00:00:00",
          "2023-09-01T00:00:00",
          "2023-10-01T00:00:00",
          "2023-11-01T00:00:00",
          "2023-12-01T00:00:00"
         ],
         "xaxis": "x",
         "y": [
          216,
          186,
          210,
          177,
          201,
          249,
          266,
          295,
          240,
          265,
          188,
          200,
          178,
          128,
          163,
          152,
          172,
          258,
          277,
          249,
          277,
          193,
          168,
          186,
          159,
          168,
          142,
          107,
          109,
          119,
          136,
          206,
          158,
          152,
          118,
          112,
          105,
          83,
          86,
          104,
          104,
          132,
          171,
          197,
          145,
          150,
          160,
          140,
          143,
          112,
          159,
          110,
          142,
          158,
          144,
          135,
          136,
          143,
          83,
          121,
          139,
          87,
          125,
          131,
          127,
          150,
          128,
          140,
          136,
          106,
          115,
          141
         ],
         "yaxis": "y"
        },
        {
         "hovertemplate": "Category=Break & Enter - Other Premises<br>Date=%{x}<br>Crime Count=%{y}<extra></extra>",
         "legendgroup": "Break & Enter - Other Premises",
         "line": {
          "color": "#ab63fa",
          "dash": "solid"
         },
         "marker": {
          "symbol": "circle"
         },
         "mode": "lines",
         "name": "Break & Enter - Other Premises",
         "orientation": "v",
         "showlegend": true,
         "type": "scatter",
         "x": [
          "2018-01-01T00:00:00",
          "2018-02-01T00:00:00",
          "2018-03-01T00:00:00",
          "2018-04-01T00:00:00",
          "2018-05-01T00:00:00",
          "2018-06-01T00:00:00",
          "2018-07-01T00:00:00",
          "2018-08-01T00:00:00",
          "2018-09-01T00:00:00",
          "2018-10-01T00:00:00",
          "2018-11-01T00:00:00",
          "2018-12-01T00:00:00",
          "2019-01-01T00:00:00",
          "2019-02-01T00:00:00",
          "2019-03-01T00:00:00",
          "2019-04-01T00:00:00",
          "2019-05-01T00:00:00",
          "2019-06-01T00:00:00",
          "2019-07-01T00:00:00",
          "2019-08-01T00:00:00",
          "2019-09-01T00:00:00",
          "2019-10-01T00:00:00",
          "2019-11-01T00:00:00",
          "2019-12-01T00:00:00",
          "2020-01-01T00:00:00",
          "2020-02-01T00:00:00",
          "2020-03-01T00:00:00",
          "2020-04-01T00:00:00",
          "2020-05-01T00:00:00",
          "2020-06-01T00:00:00",
          "2020-07-01T00:00:00",
          "2020-08-01T00:00:00",
          "2020-09-01T00:00:00",
          "2020-10-01T00:00:00",
          "2020-11-01T00:00:00",
          "2020-12-01T00:00:00",
          "2021-01-01T00:00:00",
          "2021-02-01T00:00:00",
          "2021-03-01T00:00:00",
          "2021-04-01T00:00:00",
          "2021-05-01T00:00:00",
          "2021-06-01T00:00:00",
          "2021-07-01T00:00:00",
          "2021-08-01T00:00:00",
          "2021-09-01T00:00:00",
          "2021-10-01T00:00:00",
          "2021-11-01T00:00:00",
          "2021-12-01T00:00:00",
          "2022-01-01T00:00:00",
          "2022-02-01T00:00:00",
          "2022-03-01T00:00:00",
          "2022-04-01T00:00:00",
          "2022-05-01T00:00:00",
          "2022-06-01T00:00:00",
          "2022-07-01T00:00:00",
          "2022-08-01T00:00:00",
          "2022-09-01T00:00:00",
          "2022-10-01T00:00:00",
          "2022-11-01T00:00:00",
          "2022-12-01T00:00:00",
          "2023-01-01T00:00:00",
          "2023-02-01T00:00:00",
          "2023-03-01T00:00:00",
          "2023-04-01T00:00:00",
          "2023-05-01T00:00:00",
          "2023-06-01T00:00:00",
          "2023-07-01T00:00:00",
          "2023-08-01T00:00:00",
          "2023-09-01T00:00:00",
          "2023-10-01T00:00:00",
          "2023-11-01T00:00:00",
          "2023-12-01T00:00:00"
         ],
         "xaxis": "x",
         "y": [
          133,
          115,
          106,
          184,
          213,
          222,
          215,
          197,
          213,
          191,
          167,
          147,
          174,
          142,
          168,
          192,
          202,
          214,
          219,
          259,
          215,
          190,
          158,
          170,
          179,
          180,
          179,
          203,
          147,
          222,
          217,
          301,
          299,
          220,
          199,
          154,
          151,
          87,
          133,
          133,
          136,
          159,
          175,
          177,
          192,
          176,
          168,
          148,
          170,
          156,
          199,
          202,
          200,
          189,
          150,
          155,
          150,
          146,
          113,
          118,
          99,
          73,
          75,
          114,
          123,
          128,
          138,
          163,
          97,
          95,
          127,
          66
         ],
         "yaxis": "y"
        },
        {
         "hovertemplate": "Category=Commercial Robbery<br>Date=%{x}<br>Crime Count=%{y}<extra></extra>",
         "legendgroup": "Commercial Robbery",
         "line": {
          "color": "#FFA15A",
          "dash": "solid"
         },
         "marker": {
          "symbol": "circle"
         },
         "mode": "lines",
         "name": "Commercial Robbery",
         "orientation": "v",
         "showlegend": true,
         "type": "scatter",
         "x": [
          "2018-01-01T00:00:00",
          "2018-02-01T00:00:00",
          "2018-03-01T00:00:00",
          "2018-04-01T00:00:00",
          "2018-05-01T00:00:00",
          "2018-06-01T00:00:00",
          "2018-07-01T00:00:00",
          "2018-08-01T00:00:00",
          "2018-09-01T00:00:00",
          "2018-10-01T00:00:00",
          "2018-11-01T00:00:00",
          "2018-12-01T00:00:00",
          "2019-01-01T00:00:00",
          "2019-02-01T00:00:00",
          "2019-03-01T00:00:00",
          "2019-04-01T00:00:00",
          "2019-05-01T00:00:00",
          "2019-06-01T00:00:00",
          "2019-07-01T00:00:00",
          "2019-08-01T00:00:00",
          "2019-09-01T00:00:00",
          "2019-10-01T00:00:00",
          "2019-11-01T00:00:00",
          "2019-12-01T00:00:00",
          "2020-01-01T00:00:00",
          "2020-02-01T00:00:00",
          "2020-03-01T00:00:00",
          "2020-04-01T00:00:00",
          "2020-05-01T00:00:00",
          "2020-06-01T00:00:00",
          "2020-07-01T00:00:00",
          "2020-08-01T00:00:00",
          "2020-09-01T00:00:00",
          "2020-10-01T00:00:00",
          "2020-11-01T00:00:00",
          "2020-12-01T00:00:00",
          "2021-01-01T00:00:00",
          "2021-02-01T00:00:00",
          "2021-03-01T00:00:00",
          "2021-04-01T00:00:00",
          "2021-05-01T00:00:00",
          "2021-06-01T00:00:00",
          "2021-07-01T00:00:00",
          "2021-08-01T00:00:00",
          "2021-09-01T00:00:00",
          "2021-10-01T00:00:00",
          "2021-11-01T00:00:00",
          "2021-12-01T00:00:00",
          "2022-01-01T00:00:00",
          "2022-02-01T00:00:00",
          "2022-03-01T00:00:00",
          "2022-04-01T00:00:00",
          "2022-05-01T00:00:00",
          "2022-06-01T00:00:00",
          "2022-07-01T00:00:00",
          "2022-08-01T00:00:00",
          "2022-09-01T00:00:00",
          "2022-10-01T00:00:00",
          "2022-11-01T00:00:00",
          "2022-12-01T00:00:00",
          "2023-01-01T00:00:00",
          "2023-02-01T00:00:00",
          "2023-03-01T00:00:00",
          "2023-04-01T00:00:00",
          "2023-05-01T00:00:00",
          "2023-06-01T00:00:00",
          "2023-07-01T00:00:00",
          "2023-08-01T00:00:00",
          "2023-09-01T00:00:00",
          "2023-10-01T00:00:00",
          "2023-11-01T00:00:00",
          "2023-12-01T00:00:00"
         ],
         "xaxis": "x",
         "y": [
          25,
          24,
          22,
          21,
          33,
          31,
          25,
          22,
          24,
          26,
          33,
          19,
          24,
          32,
          24,
          20,
          21,
          28,
          32,
          20,
          34,
          43,
          52,
          52,
          31,
          32,
          31,
          13,
          10,
          12,
          11,
          27,
          35,
          19,
          30,
          30,
          34,
          30,
          25,
          19,
          18,
          29,
          25,
          39,
          25,
          33,
          33,
          49,
          46,
          43,
          51,
          34,
          25,
          41,
          19,
          22,
          34,
          37,
          21,
          18,
          23,
          21,
          28,
          19,
          22,
          30,
          24,
          23,
          18,
          25,
          41,
          37
         ],
         "yaxis": "y"
        },
        {
         "hovertemplate": "Category=Street Robbery<br>Date=%{x}<br>Crime Count=%{y}<extra></extra>",
         "legendgroup": "Street Robbery",
         "line": {
          "color": "#19d3f3",
          "dash": "solid"
         },
         "marker": {
          "symbol": "circle"
         },
         "mode": "lines",
         "name": "Street Robbery",
         "orientation": "v",
         "showlegend": true,
         "type": "scatter",
         "x": [
          "2018-01-01T00:00:00",
          "2018-02-01T00:00:00",
          "2018-03-01T00:00:00",
          "2018-04-01T00:00:00",
          "2018-05-01T00:00:00",
          "2018-06-01T00:00:00",
          "2018-07-01T00:00:00",
          "2018-08-01T00:00:00",
          "2018-09-01T00:00:00",
          "2018-10-01T00:00:00",
          "2018-11-01T00:00:00",
          "2018-12-01T00:00:00",
          "2019-01-01T00:00:00",
          "2019-02-01T00:00:00",
          "2019-03-01T00:00:00",
          "2019-04-01T00:00:00",
          "2019-05-01T00:00:00",
          "2019-06-01T00:00:00",
          "2019-07-01T00:00:00",
          "2019-08-01T00:00:00",
          "2019-09-01T00:00:00",
          "2019-10-01T00:00:00",
          "2019-11-01T00:00:00",
          "2019-12-01T00:00:00",
          "2020-01-01T00:00:00",
          "2020-02-01T00:00:00",
          "2020-03-01T00:00:00",
          "2020-04-01T00:00:00",
          "2020-05-01T00:00:00",
          "2020-06-01T00:00:00",
          "2020-07-01T00:00:00",
          "2020-08-01T00:00:00",
          "2020-09-01T00:00:00",
          "2020-10-01T00:00:00",
          "2020-11-01T00:00:00",
          "2020-12-01T00:00:00",
          "2021-01-01T00:00:00",
          "2021-02-01T00:00:00",
          "2021-03-01T00:00:00",
          "2021-04-01T00:00:00",
          "2021-05-01T00:00:00",
          "2021-06-01T00:00:00",
          "2021-07-01T00:00:00",
          "2021-08-01T00:00:00",
          "2021-09-01T00:00:00",
          "2021-10-01T00:00:00",
          "2021-11-01T00:00:00",
          "2021-12-01T00:00:00",
          "2022-01-01T00:00:00",
          "2022-02-01T00:00:00",
          "2022-03-01T00:00:00",
          "2022-04-01T00:00:00",
          "2022-05-01T00:00:00",
          "2022-06-01T00:00:00",
          "2022-07-01T00:00:00",
          "2022-08-01T00:00:00",
          "2022-09-01T00:00:00",
          "2022-10-01T00:00:00",
          "2022-11-01T00:00:00",
          "2022-12-01T00:00:00",
          "2023-01-01T00:00:00",
          "2023-02-01T00:00:00",
          "2023-03-01T00:00:00",
          "2023-04-01T00:00:00",
          "2023-05-01T00:00:00",
          "2023-06-01T00:00:00",
          "2023-07-01T00:00:00",
          "2023-08-01T00:00:00",
          "2023-09-01T00:00:00",
          "2023-10-01T00:00:00",
          "2023-11-01T00:00:00",
          "2023-12-01T00:00:00"
         ],
         "xaxis": "x",
         "y": [
          38,
          35,
          39,
          43,
          71,
          52,
          61,
          63,
          61,
          62,
          70,
          63,
          64,
          20,
          54,
          52,
          49,
          45,
          64,
          58,
          73,
          71,
          51,
          48,
          66,
          59,
          52,
          28,
          32,
          23,
          40,
          59,
          42,
          43,
          46,
          35,
          32,
          43,
          36,
          53,
          63,
          46,
          63,
          48,
          45,
          52,
          60,
          34,
          51,
          40,
          38,
          33,
          49,
          33,
          37,
          61,
          65,
          58,
          38,
          33,
          43,
          45,
          75,
          55,
          71,
          75,
          60,
          62,
          57,
          54,
          49,
          47
         ],
         "yaxis": "y"
        },
        {
         "hovertemplate": "Category=Theft FROM Vehicle<br>Date=%{x}<br>Crime Count=%{y}<extra></extra>",
         "legendgroup": "Theft FROM Vehicle",
         "line": {
          "color": "#FF6692",
          "dash": "solid"
         },
         "marker": {
          "symbol": "circle"
         },
         "mode": "lines",
         "name": "Theft FROM Vehicle",
         "orientation": "v",
         "showlegend": true,
         "type": "scatter",
         "x": [
          "2018-01-01T00:00:00",
          "2018-02-01T00:00:00",
          "2018-03-01T00:00:00",
          "2018-04-01T00:00:00",
          "2018-05-01T00:00:00",
          "2018-06-01T00:00:00",
          "2018-07-01T00:00:00",
          "2018-08-01T00:00:00",
          "2018-09-01T00:00:00",
          "2018-10-01T00:00:00",
          "2018-11-01T00:00:00",
          "2018-12-01T00:00:00",
          "2019-01-01T00:00:00",
          "2019-02-01T00:00:00",
          "2019-03-01T00:00:00",
          "2019-04-01T00:00:00",
          "2019-05-01T00:00:00",
          "2019-06-01T00:00:00",
          "2019-07-01T00:00:00",
          "2019-08-01T00:00:00",
          "2019-09-01T00:00:00",
          "2019-10-01T00:00:00",
          "2019-11-01T00:00:00",
          "2019-12-01T00:00:00",
          "2020-01-01T00:00:00",
          "2020-02-01T00:00:00",
          "2020-03-01T00:00:00",
          "2020-04-01T00:00:00",
          "2020-05-01T00:00:00",
          "2020-06-01T00:00:00",
          "2020-07-01T00:00:00",
          "2020-08-01T00:00:00",
          "2020-09-01T00:00:00",
          "2020-10-01T00:00:00",
          "2020-11-01T00:00:00",
          "2020-12-01T00:00:00",
          "2021-01-01T00:00:00",
          "2021-02-01T00:00:00",
          "2021-03-01T00:00:00",
          "2021-04-01T00:00:00",
          "2021-05-01T00:00:00",
          "2021-06-01T00:00:00",
          "2021-07-01T00:00:00",
          "2021-08-01T00:00:00",
          "2021-09-01T00:00:00",
          "2021-10-01T00:00:00",
          "2021-11-01T00:00:00",
          "2021-12-01T00:00:00",
          "2022-01-01T00:00:00",
          "2022-02-01T00:00:00",
          "2022-03-01T00:00:00",
          "2022-04-01T00:00:00",
          "2022-05-01T00:00:00",
          "2022-06-01T00:00:00",
          "2022-07-01T00:00:00",
          "2022-08-01T00:00:00",
          "2022-09-01T00:00:00",
          "2022-10-01T00:00:00",
          "2022-11-01T00:00:00",
          "2022-12-01T00:00:00",
          "2023-01-01T00:00:00",
          "2023-02-01T00:00:00",
          "2023-03-01T00:00:00",
          "2023-04-01T00:00:00",
          "2023-05-01T00:00:00",
          "2023-06-01T00:00:00",
          "2023-07-01T00:00:00",
          "2023-08-01T00:00:00",
          "2023-09-01T00:00:00",
          "2023-10-01T00:00:00",
          "2023-11-01T00:00:00",
          "2023-12-01T00:00:00"
         ],
         "xaxis": "x",
         "y": [
          959,
          638,
          815,
          870,
          1059,
          1033,
          1315,
          1399,
          1289,
          1267,
          1027,
          953,
          1094,
          713,
          917,
          1093,
          1265,
          1342,
          1519,
          1613,
          1507,
          1310,
          1250,
          1100,
          1169,
          1203,
          1080,
          1067,
          793,
          821,
          1072,
          1187,
          1219,
          1220,
          877,
          679,
          818,
          615,
          846,
          819,
          844,
          778,
          901,
          962,
          909,
          943,
          937,
          786,
          1055,
          895,
          1194,
          1062,
          1144,
          1212,
          1090,
          1134,
          971,
          935,
          838,
          814,
          859,
          754,
          866,
          808,
          879,
          896,
          842,
          734,
          686,
          592,
          589,
          446
         ],
         "yaxis": "y"
        },
        {
         "hovertemplate": "Category=Theft OF Vehicle<br>Date=%{x}<br>Crime Count=%{y}<extra></extra>",
         "legendgroup": "Theft OF Vehicle",
         "line": {
          "color": "#B6E880",
          "dash": "solid"
         },
         "marker": {
          "symbol": "circle"
         },
         "mode": "lines",
         "name": "Theft OF Vehicle",
         "orientation": "v",
         "showlegend": true,
         "type": "scatter",
         "x": [
          "2018-01-01T00:00:00",
          "2018-02-01T00:00:00",
          "2018-03-01T00:00:00",
          "2018-04-01T00:00:00",
          "2018-05-01T00:00:00",
          "2018-06-01T00:00:00",
          "2018-07-01T00:00:00",
          "2018-08-01T00:00:00",
          "2018-09-01T00:00:00",
          "2018-10-01T00:00:00",
          "2018-11-01T00:00:00",
          "2018-12-01T00:00:00",
          "2019-01-01T00:00:00",
          "2019-02-01T00:00:00",
          "2019-03-01T00:00:00",
          "2019-04-01T00:00:00",
          "2019-05-01T00:00:00",
          "2019-06-01T00:00:00",
          "2019-07-01T00:00:00",
          "2019-08-01T00:00:00",
          "2019-09-01T00:00:00",
          "2019-10-01T00:00:00",
          "2019-11-01T00:00:00",
          "2019-12-01T00:00:00",
          "2020-01-01T00:00:00",
          "2020-02-01T00:00:00",
          "2020-03-01T00:00:00",
          "2020-04-01T00:00:00",
          "2020-05-01T00:00:00",
          "2020-06-01T00:00:00",
          "2020-07-01T00:00:00",
          "2020-08-01T00:00:00",
          "2020-09-01T00:00:00",
          "2020-10-01T00:00:00",
          "2020-11-01T00:00:00",
          "2020-12-01T00:00:00",
          "2021-01-01T00:00:00",
          "2021-02-01T00:00:00",
          "2021-03-01T00:00:00",
          "2021-04-01T00:00:00",
          "2021-05-01T00:00:00",
          "2021-06-01T00:00:00",
          "2021-07-01T00:00:00",
          "2021-08-01T00:00:00",
          "2021-09-01T00:00:00",
          "2021-10-01T00:00:00",
          "2021-11-01T00:00:00",
          "2021-12-01T00:00:00",
          "2022-01-01T00:00:00",
          "2022-02-01T00:00:00",
          "2022-03-01T00:00:00",
          "2022-04-01T00:00:00",
          "2022-05-01T00:00:00",
          "2022-06-01T00:00:00",
          "2022-07-01T00:00:00",
          "2022-08-01T00:00:00",
          "2022-09-01T00:00:00",
          "2022-10-01T00:00:00",
          "2022-11-01T00:00:00",
          "2022-12-01T00:00:00",
          "2023-01-01T00:00:00",
          "2023-02-01T00:00:00",
          "2023-03-01T00:00:00",
          "2023-04-01T00:00:00",
          "2023-05-01T00:00:00",
          "2023-06-01T00:00:00",
          "2023-07-01T00:00:00",
          "2023-08-01T00:00:00",
          "2023-09-01T00:00:00",
          "2023-10-01T00:00:00",
          "2023-11-01T00:00:00",
          "2023-12-01T00:00:00"
         ],
         "xaxis": "x",
         "y": [
          600,
          550,
          501,
          513,
          455,
          419,
          591,
          503,
          492,
          478,
          455,
          499,
          460,
          465,
          433,
          440,
          483,
          464,
          625,
          561,
          647,
          507,
          505,
          438,
          509,
          441,
          468,
          420,
          259,
          295,
          354,
          379,
          408,
          409,
          350,
          357,
          437,
          378,
          319,
          351,
          314,
          323,
          391,
          371,
          369,
          351,
          385,
          456,
          477,
          456,
          542,
          432,
          464,
          447,
          441,
          532,
          351,
          429,
          396,
          452,
          423,
          343,
          437,
          323,
          369,
          379,
          375,
          353,
          401,
          348,
          335,
          362
         ],
         "yaxis": "y"
        },
        {
         "hovertemplate": "Category=Violence Other (Non-domestic)<br>Date=%{x}<br>Crime Count=%{y}<extra></extra>",
         "legendgroup": "Violence Other (Non-domestic)",
         "line": {
          "color": "#FF97FF",
          "dash": "solid"
         },
         "marker": {
          "symbol": "circle"
         },
         "mode": "lines",
         "name": "Violence Other (Non-domestic)",
         "orientation": "v",
         "showlegend": true,
         "type": "scatter",
         "x": [
          "2018-01-01T00:00:00",
          "2018-02-01T00:00:00",
          "2018-03-01T00:00:00",
          "2018-04-01T00:00:00",
          "2018-05-01T00:00:00",
          "2018-06-01T00:00:00",
          "2018-07-01T00:00:00",
          "2018-08-01T00:00:00",
          "2018-09-01T00:00:00",
          "2018-10-01T00:00:00",
          "2018-11-01T00:00:00",
          "2018-12-01T00:00:00",
          "2019-01-01T00:00:00",
          "2019-02-01T00:00:00",
          "2019-03-01T00:00:00",
          "2019-04-01T00:00:00",
          "2019-05-01T00:00:00",
          "2019-06-01T00:00:00",
          "2019-07-01T00:00:00",
          "2019-08-01T00:00:00",
          "2019-09-01T00:00:00",
          "2019-10-01T00:00:00",
          "2019-11-01T00:00:00",
          "2019-12-01T00:00:00",
          "2020-01-01T00:00:00",
          "2020-02-01T00:00:00",
          "2020-03-01T00:00:00",
          "2020-04-01T00:00:00",
          "2020-05-01T00:00:00",
          "2020-06-01T00:00:00",
          "2020-07-01T00:00:00",
          "2020-08-01T00:00:00",
          "2020-09-01T00:00:00",
          "2020-10-01T00:00:00",
          "2020-11-01T00:00:00",
          "2020-12-01T00:00:00",
          "2021-01-01T00:00:00",
          "2021-02-01T00:00:00",
          "2021-03-01T00:00:00",
          "2021-04-01T00:00:00",
          "2021-05-01T00:00:00",
          "2021-06-01T00:00:00",
          "2021-07-01T00:00:00",
          "2021-08-01T00:00:00",
          "2021-09-01T00:00:00",
          "2021-10-01T00:00:00",
          "2021-11-01T00:00:00",
          "2021-12-01T00:00:00",
          "2022-01-01T00:00:00",
          "2022-02-01T00:00:00",
          "2022-03-01T00:00:00",
          "2022-04-01T00:00:00",
          "2022-05-01T00:00:00",
          "2022-06-01T00:00:00",
          "2022-07-01T00:00:00",
          "2022-08-01T00:00:00",
          "2022-09-01T00:00:00",
          "2022-10-01T00:00:00",
          "2022-11-01T00:00:00",
          "2022-12-01T00:00:00",
          "2023-01-01T00:00:00",
          "2023-02-01T00:00:00",
          "2023-03-01T00:00:00",
          "2023-04-01T00:00:00",
          "2023-05-01T00:00:00",
          "2023-06-01T00:00:00",
          "2023-07-01T00:00:00",
          "2023-08-01T00:00:00",
          "2023-09-01T00:00:00",
          "2023-10-01T00:00:00",
          "2023-11-01T00:00:00",
          "2023-12-01T00:00:00"
         ],
         "xaxis": "x",
         "y": [
          190,
          146,
          175,
          154,
          175,
          188,
          158,
          182,
          154,
          174,
          165,
          150,
          204,
          153,
          158,
          147,
          169,
          163,
          180,
          146,
          163,
          158,
          180,
          152,
          196,
          159,
          146,
          124,
          138,
          182,
          203,
          192,
          191,
          182,
          166,
          135,
          182,
          131,
          177,
          164,
          169,
          190,
          181,
          195,
          220,
          186,
          198,
          186,
          146,
          172,
          174,
          170,
          168,
          156,
          218,
          156,
          184,
          208,
          158,
          174,
          174,
          168,
          193,
          185,
          237,
          199,
          203,
          186,
          168,
          155,
          139,
          122
         ],
         "yaxis": "y"
        }
       ],
       "layout": {
        "height": 1000,
        "legend": {
         "title": {
          "text": "Category"
         },
         "tracegroupgap": 0
        },
        "template": {
         "data": {
          "bar": [
           {
            "error_x": {
             "color": "#2a3f5f"
            },
            "error_y": {
             "color": "#2a3f5f"
            },
            "marker": {
             "line": {
              "color": "#E5ECF6",
              "width": 0.5
             },
             "pattern": {
              "fillmode": "overlay",
              "size": 10,
              "solidity": 0.2
             }
            },
            "type": "bar"
           }
          ],
          "barpolar": [
           {
            "marker": {
             "line": {
              "color": "#E5ECF6",
              "width": 0.5
             },
             "pattern": {
              "fillmode": "overlay",
              "size": 10,
              "solidity": 0.2
             }
            },
            "type": "barpolar"
           }
          ],
          "carpet": [
           {
            "aaxis": {
             "endlinecolor": "#2a3f5f",
             "gridcolor": "white",
             "linecolor": "white",
             "minorgridcolor": "white",
             "startlinecolor": "#2a3f5f"
            },
            "baxis": {
             "endlinecolor": "#2a3f5f",
             "gridcolor": "white",
             "linecolor": "white",
             "minorgridcolor": "white",
             "startlinecolor": "#2a3f5f"
            },
            "type": "carpet"
           }
          ],
          "choropleth": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "type": "choropleth"
           }
          ],
          "contour": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "contour"
           }
          ],
          "contourcarpet": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "type": "contourcarpet"
           }
          ],
          "heatmap": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "heatmap"
           }
          ],
          "heatmapgl": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "heatmapgl"
           }
          ],
          "histogram": [
           {
            "marker": {
             "pattern": {
              "fillmode": "overlay",
              "size": 10,
              "solidity": 0.2
             }
            },
            "type": "histogram"
           }
          ],
          "histogram2d": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "histogram2d"
           }
          ],
          "histogram2dcontour": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "histogram2dcontour"
           }
          ],
          "mesh3d": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "type": "mesh3d"
           }
          ],
          "parcoords": [
           {
            "line": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "parcoords"
           }
          ],
          "pie": [
           {
            "automargin": true,
            "type": "pie"
           }
          ],
          "scatter": [
           {
            "fillpattern": {
             "fillmode": "overlay",
             "size": 10,
             "solidity": 0.2
            },
            "type": "scatter"
           }
          ],
          "scatter3d": [
           {
            "line": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatter3d"
           }
          ],
          "scattercarpet": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattercarpet"
           }
          ],
          "scattergeo": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattergeo"
           }
          ],
          "scattergl": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattergl"
           }
          ],
          "scattermapbox": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattermapbox"
           }
          ],
          "scatterpolar": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatterpolar"
           }
          ],
          "scatterpolargl": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatterpolargl"
           }
          ],
          "scatterternary": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatterternary"
           }
          ],
          "surface": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "surface"
           }
          ],
          "table": [
           {
            "cells": {
             "fill": {
              "color": "#EBF0F8"
             },
             "line": {
              "color": "white"
             }
            },
            "header": {
             "fill": {
              "color": "#C8D4E3"
             },
             "line": {
              "color": "white"
             }
            },
            "type": "table"
           }
          ]
         },
         "layout": {
          "annotationdefaults": {
           "arrowcolor": "#2a3f5f",
           "arrowhead": 0,
           "arrowwidth": 1
          },
          "autotypenumbers": "strict",
          "coloraxis": {
           "colorbar": {
            "outlinewidth": 0,
            "ticks": ""
           }
          },
          "colorscale": {
           "diverging": [
            [
             0,
             "#8e0152"
            ],
            [
             0.1,
             "#c51b7d"
            ],
            [
             0.2,
             "#de77ae"
            ],
            [
             0.3,
             "#f1b6da"
            ],
            [
             0.4,
             "#fde0ef"
            ],
            [
             0.5,
             "#f7f7f7"
            ],
            [
             0.6,
             "#e6f5d0"
            ],
            [
             0.7,
             "#b8e186"
            ],
            [
             0.8,
             "#7fbc41"
            ],
            [
             0.9,
             "#4d9221"
            ],
            [
             1,
             "#276419"
            ]
           ],
           "sequential": [
            [
             0,
             "#0d0887"
            ],
            [
             0.1111111111111111,
             "#46039f"
            ],
            [
             0.2222222222222222,
             "#7201a8"
            ],
            [
             0.3333333333333333,
             "#9c179e"
            ],
            [
             0.4444444444444444,
             "#bd3786"
            ],
            [
             0.5555555555555556,
             "#d8576b"
            ],
            [
             0.6666666666666666,
             "#ed7953"
            ],
            [
             0.7777777777777778,
             "#fb9f3a"
            ],
            [
             0.8888888888888888,
             "#fdca26"
            ],
            [
             1,
             "#f0f921"
            ]
           ],
           "sequentialminus": [
            [
             0,
             "#0d0887"
            ],
            [
             0.1111111111111111,
             "#46039f"
            ],
            [
             0.2222222222222222,
             "#7201a8"
            ],
            [
             0.3333333333333333,
             "#9c179e"
            ],
            [
             0.4444444444444444,
             "#bd3786"
            ],
            [
             0.5555555555555556,
             "#d8576b"
            ],
            [
             0.6666666666666666,
             "#ed7953"
            ],
            [
             0.7777777777777778,
             "#fb9f3a"
            ],
            [
             0.8888888888888888,
             "#fdca26"
            ],
            [
             1,
             "#f0f921"
            ]
           ]
          },
          "colorway": [
           "#636efa",
           "#EF553B",
           "#00cc96",
           "#ab63fa",
           "#FFA15A",
           "#19d3f3",
           "#FF6692",
           "#B6E880",
           "#FF97FF",
           "#FECB52"
          ],
          "font": {
           "color": "#2a3f5f"
          },
          "geo": {
           "bgcolor": "white",
           "lakecolor": "white",
           "landcolor": "#E5ECF6",
           "showlakes": true,
           "showland": true,
           "subunitcolor": "white"
          },
          "hoverlabel": {
           "align": "left"
          },
          "hovermode": "closest",
          "mapbox": {
           "style": "light"
          },
          "paper_bgcolor": "white",
          "plot_bgcolor": "#E5ECF6",
          "polar": {
           "angularaxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           },
           "bgcolor": "#E5ECF6",
           "radialaxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           }
          },
          "scene": {
           "xaxis": {
            "backgroundcolor": "#E5ECF6",
            "gridcolor": "white",
            "gridwidth": 2,
            "linecolor": "white",
            "showbackground": true,
            "ticks": "",
            "zerolinecolor": "white"
           },
           "yaxis": {
            "backgroundcolor": "#E5ECF6",
            "gridcolor": "white",
            "gridwidth": 2,
            "linecolor": "white",
            "showbackground": true,
            "ticks": "",
            "zerolinecolor": "white"
           },
           "zaxis": {
            "backgroundcolor": "#E5ECF6",
            "gridcolor": "white",
            "gridwidth": 2,
            "linecolor": "white",
            "showbackground": true,
            "ticks": "",
            "zerolinecolor": "white"
           }
          },
          "shapedefaults": {
           "line": {
            "color": "#2a3f5f"
           }
          },
          "ternary": {
           "aaxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           },
           "baxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           },
           "bgcolor": "#E5ECF6",
           "caxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           }
          },
          "title": {
           "x": 0.05
          },
          "xaxis": {
           "automargin": true,
           "gridcolor": "white",
           "linecolor": "white",
           "ticks": "",
           "title": {
            "standoff": 15
           },
           "zerolinecolor": "white",
           "zerolinewidth": 2
          },
          "yaxis": {
           "automargin": true,
           "gridcolor": "white",
           "linecolor": "white",
           "ticks": "",
           "title": {
            "standoff": 15
           },
           "zerolinecolor": "white",
           "zerolinewidth": 2
          }
         }
        },
        "title": {
         "text": "Crime Count Over Time by Type"
        },
        "width": 2000,
        "xaxis": {
         "anchor": "y",
         "domain": [
          0,
          1
         ],
         "title": {
          "text": "Date"
         }
        },
        "yaxis": {
         "anchor": "x",
         "domain": [
          0,
          1
         ],
         "title": {
          "text": "Crime count"
         }
        }
       }
      }
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "import pandas as pd\n",
    "import plotly.express as px\n",
    "\n",
    "df_grouped = df_clean.groupby(['Category', 'Date'])['Crime Count'].sum().reset_index()\n",
    "fig = px.line(df_grouped, x='Date', y='Crime Count', color='Category', title='Crime Count Over Time by Type')\n",
    "fig.update_layout(xaxis_title='Date', yaxis_title='Crime count', legend_title='Category', width=2000, height=1000)\n",
    "fig.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "In this graph we can also see that the crimes have remained the same, with the exception of 'theft from vehicle', which is the category with the highest number of crimes historically. We can see that this category has been decreasing over time"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Crime Rate per Capita"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "To solve this question, I will create a tree map to show wich community of the city have the highest number of crimes per 1000 people"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.read_csv('Data_2.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/vnd.plotly.v1+json": {
       "config": {
        "plotlyServerURL": "https://plot.ly"
       },
       "data": [
        {
         "branchvalues": "total",
         "domain": {
          "x": [
           0,
           1
          ],
          "y": [
           0,
           1
          ]
         },
         "hovertemplate": "labels=%{label}<br>crime_rate_per_1000=%{value}<br>parent=%{parent}<br>id=%{id}<extra></extra>",
         "ids": [
          "ABBEYDALE",
          "ACADIA",
          "ALBERT PARK/RADISSON HEIGHTS",
          "ALTADORE",
          "APPLEWOOD PARK",
          "ARBOUR LAKE",
          "ASPEN WOODS",
          "AUBURN BAY",
          "BANFF TRAIL",
          "BANKVIEW",
          "BAYVIEW",
          "BEDDINGTON HEIGHTS",
          "BEL-AIRE",
          "BELMONT",
          "BELTLINE",
          "BELVEDERE",
          "BONAVISTA DOWNS",
          "BOWNESS",
          "BRAESIDE",
          "BRENTWOOD",
          "BRIDGELAND/RIVERSIDE",
          "BRIDLEWOOD",
          "BRITANNIA",
          "CAMBRIAN HEIGHTS",
          "CANYON MEADOWS",
          "CAPITOL HILL",
          "CARRINGTON",
          "CASTLERIDGE",
          "CEDARBRAE",
          "CHAPARRAL",
          "CHARLESWOOD",
          "CHINATOWN",
          "CHINOOK PARK",
          "CHRISTIE PARK",
          "CITADEL",
          "CITYSCAPE",
          "CLIFF BUNGALOW",
          "COACH HILL",
          "COLLINGWOOD",
          "COPPERFIELD",
          "CORAL SPRINGS",
          "CORNERSTONE",
          "COUGAR RIDGE",
          "COUNTRY HILLS",
          "COUNTRY HILLS VILLAGE",
          "COVENTRY HILLS",
          "CRANSTON",
          "CRESCENT HEIGHTS",
          "CRESTMONT",
          "CURRIE BARRACKS",
          "DALHOUSIE",
          "DEER RIDGE",
          "DEER RUN",
          "DIAMOND COVE",
          "DISCOVERY RIDGE",
          "DOUGLASDALE/GLEN",
          "DOVER",
          "DOWNTOWN COMMERCIAL CORE",
          "DOWNTOWN EAST VILLAGE",
          "DOWNTOWN WEST END",
          "EAGLE RIDGE",
          "EAU CLAIRE",
          "EDGEMONT",
          "ELBOW PARK",
          "ELBOYA",
          "ERIN WOODS",
          "ERLTON",
          "EVANSTON",
          "EVERGREEN",
          "FAIRVIEW",
          "FALCONRIDGE",
          "FOOTHILLS",
          "FOREST HEIGHTS",
          "FOREST LAWN",
          "FOREST LAWN INDUSTRIAL",
          "GARRISON GREEN",
          "GARRISON WOODS",
          "GLAMORGAN",
          "GLENBROOK",
          "GLENDALE",
          "GREENVIEW",
          "GREENVIEW INDUSTRIAL PARK",
          "GREENWOOD/GREENBRIAR",
          "HAMPTONS",
          "HARVEST HILLS",
          "HAWKWOOD",
          "HAYSBORO",
          "HIDDEN VALLEY",
          "HIGHLAND PARK",
          "HIGHWOOD",
          "HILLHURST",
          "HOUNSFIELD HEIGHTS/BRIAR HILL",
          "HUNTINGTON HILLS",
          "INGLEWOOD",
          "KELVIN GROVE",
          "KILLARNEY/GLENGARRY",
          "KINCORA",
          "KINGSLAND",
          "LAKE BONAVISTA",
          "LAKEVIEW",
          "LEGACY",
          "LINCOLN PARK",
          "LIVINGSTON",
          "LOWER MOUNT ROYAL",
          "MACEWAN GLEN",
          "MAHOGANY",
          "MANCHESTER",
          "MAPLE RIDGE",
          "MARLBOROUGH",
          "MARLBOROUGH PARK",
          "MARTINDALE",
          "MAYFAIR",
          "MAYLAND HEIGHTS",
          "MCKENZIE LAKE",
          "MCKENZIE TOWNE",
          "MEADOWLARK PARK",
          "MIDNAPORE",
          "MILLRISE",
          "MISSION",
          "MONTEREY PARK",
          "MONTGOMERY",
          "MOUNT PLEASANT",
          "NEW BRIGHTON",
          "NOLAN HILL",
          "NORTH GLENMORE PARK",
          "NORTH HAVEN",
          "NORTH HAVEN UPPER",
          "OAKRIDGE",
          "OGDEN",
          "PALLISER",
          "PANORAMA HILLS",
          "PARKDALE",
          "PARKHILL",
          "PARKLAND",
          "PATTERSON",
          "PENBROOKE MEADOWS",
          "PINE CREEK",
          "PINERIDGE",
          "POINT MCKAY",
          "PUMP HILL",
          "QUEENS PARK VILLAGE",
          "QUEENSLAND",
          "RAMSAY",
          "RANCHLANDS",
          "RED CARPET",
          "REDSTONE",
          "RENFREW",
          "RICHMOND",
          "RIDEAU PARK",
          "RIVERBEND",
          "ROCKY RIDGE",
          "ROSEDALE",
          "ROSEMONT",
          "ROSSCARROCK",
          "ROXBORO",
          "ROYAL OAK",
          "RUNDLE",
          "RUTLAND PARK",
          "SADDLE RIDGE",
          "SADDLE RIDGE INDUSTRIAL",
          "SAGE HILL",
          "SANDSTONE VALLEY",
          "SCARBORO",
          "SCARBORO/ SUNALTA WEST",
          "SCENIC ACRES",
          "SETON",
          "SHAGANAPPI",
          "SHAWNEE SLOPES",
          "SHAWNESSY",
          "SHEPARD INDUSTRIAL",
          "SHERWOOD",
          "SIGNAL HILL",
          "SILVER SPRINGS",
          "SILVERADO",
          "SKYVIEW RANCH",
          "SOMERSET",
          "SOUTH CALGARY",
          "SOUTHVIEW",
          "SOUTHWOOD",
          "SPRINGBANK HILL",
          "SPRUCE CLIFF",
          "ST. ANDREWS HEIGHTS",
          "STRATHCONA PARK",
          "SUNALTA",
          "SUNDANCE",
          "SUNNYSIDE",
          "SUNRIDGE",
          "TARADALE",
          "TEMPLE",
          "THORNCLIFFE",
          "TUSCANY",
          "TUXEDO PARK",
          "UNIVERSITY DISTRICT",
          "UNIVERSITY HEIGHTS",
          "UNIVERSITY OF CALGARY",
          "UPPER MOUNT ROYAL",
          "VALLEY RIDGE",
          "VARSITY",
          "VISTA HEIGHTS",
          "WALDEN",
          "WEST HILLHURST",
          "WEST SPRINGS",
          "WESTGATE",
          "WHITEHORN",
          "WILDWOOD",
          "WILLOW PARK",
          "WINDSOR PARK",
          "WINSTON HEIGHTS/MOUNTVIEW",
          "WOODBINE",
          "WOODLANDS",
          "YORKVILLE"
         ],
         "labels": [
          "ABBEYDALE",
          "ACADIA",
          "ALBERT PARK/RADISSON HEIGHTS",
          "ALTADORE",
          "APPLEWOOD PARK",
          "ARBOUR LAKE",
          "ASPEN WOODS",
          "AUBURN BAY",
          "BANFF TRAIL",
          "BANKVIEW",
          "BAYVIEW",
          "BEDDINGTON HEIGHTS",
          "BEL-AIRE",
          "BELMONT",
          "BELTLINE",
          "BELVEDERE",
          "BONAVISTA DOWNS",
          "BOWNESS",
          "BRAESIDE",
          "BRENTWOOD",
          "BRIDGELAND/RIVERSIDE",
          "BRIDLEWOOD",
          "BRITANNIA",
          "CAMBRIAN HEIGHTS",
          "CANYON MEADOWS",
          "CAPITOL HILL",
          "CARRINGTON",
          "CASTLERIDGE",
          "CEDARBRAE",
          "CHAPARRAL",
          "CHARLESWOOD",
          "CHINATOWN",
          "CHINOOK PARK",
          "CHRISTIE PARK",
          "CITADEL",
          "CITYSCAPE",
          "CLIFF BUNGALOW",
          "COACH HILL",
          "COLLINGWOOD",
          "COPPERFIELD",
          "CORAL SPRINGS",
          "CORNERSTONE",
          "COUGAR RIDGE",
          "COUNTRY HILLS",
          "COUNTRY HILLS VILLAGE",
          "COVENTRY HILLS",
          "CRANSTON",
          "CRESCENT HEIGHTS",
          "CRESTMONT",
          "CURRIE BARRACKS",
          "DALHOUSIE",
          "DEER RIDGE",
          "DEER RUN",
          "DIAMOND COVE",
          "DISCOVERY RIDGE",
          "DOUGLASDALE/GLEN",
          "DOVER",
          "DOWNTOWN COMMERCIAL CORE",
          "DOWNTOWN EAST VILLAGE",
          "DOWNTOWN WEST END",
          "EAGLE RIDGE",
          "EAU CLAIRE",
          "EDGEMONT",
          "ELBOW PARK",
          "ELBOYA",
          "ERIN WOODS",
          "ERLTON",
          "EVANSTON",
          "EVERGREEN",
          "FAIRVIEW",
          "FALCONRIDGE",
          "FOOTHILLS",
          "FOREST HEIGHTS",
          "FOREST LAWN",
          "FOREST LAWN INDUSTRIAL",
          "GARRISON GREEN",
          "GARRISON WOODS",
          "GLAMORGAN",
          "GLENBROOK",
          "GLENDALE",
          "GREENVIEW",
          "GREENVIEW INDUSTRIAL PARK",
          "GREENWOOD/GREENBRIAR",
          "HAMPTONS",
          "HARVEST HILLS",
          "HAWKWOOD",
          "HAYSBORO",
          "HIDDEN VALLEY",
          "HIGHLAND PARK",
          "HIGHWOOD",
          "HILLHURST",
          "HOUNSFIELD HEIGHTS/BRIAR HILL",
          "HUNTINGTON HILLS",
          "INGLEWOOD",
          "KELVIN GROVE",
          "KILLARNEY/GLENGARRY",
          "KINCORA",
          "KINGSLAND",
          "LAKE BONAVISTA",
          "LAKEVIEW",
          "LEGACY",
          "LINCOLN PARK",
          "LIVINGSTON",
          "LOWER MOUNT ROYAL",
          "MACEWAN GLEN",
          "MAHOGANY",
          "MANCHESTER",
          "MAPLE RIDGE",
          "MARLBOROUGH",
          "MARLBOROUGH PARK",
          "MARTINDALE",
          "MAYFAIR",
          "MAYLAND HEIGHTS",
          "MCKENZIE LAKE",
          "MCKENZIE TOWNE",
          "MEADOWLARK PARK",
          "MIDNAPORE",
          "MILLRISE",
          "MISSION",
          "MONTEREY PARK",
          "MONTGOMERY",
          "MOUNT PLEASANT",
          "NEW BRIGHTON",
          "NOLAN HILL",
          "NORTH GLENMORE PARK",
          "NORTH HAVEN",
          "NORTH HAVEN UPPER",
          "OAKRIDGE",
          "OGDEN",
          "PALLISER",
          "PANORAMA HILLS",
          "PARKDALE",
          "PARKHILL",
          "PARKLAND",
          "PATTERSON",
          "PENBROOKE MEADOWS",
          "PINE CREEK",
          "PINERIDGE",
          "POINT MCKAY",
          "PUMP HILL",
          "QUEENS PARK VILLAGE",
          "QUEENSLAND",
          "RAMSAY",
          "RANCHLANDS",
          "RED CARPET",
          "REDSTONE",
          "RENFREW",
          "RICHMOND",
          "RIDEAU PARK",
          "RIVERBEND",
          "ROCKY RIDGE",
          "ROSEDALE",
          "ROSEMONT",
          "ROSSCARROCK",
          "ROXBORO",
          "ROYAL OAK",
          "RUNDLE",
          "RUTLAND PARK",
          "SADDLE RIDGE",
          "SADDLE RIDGE INDUSTRIAL",
          "SAGE HILL",
          "SANDSTONE VALLEY",
          "SCARBORO",
          "SCARBORO/ SUNALTA WEST",
          "SCENIC ACRES",
          "SETON",
          "SHAGANAPPI",
          "SHAWNEE SLOPES",
          "SHAWNESSY",
          "SHEPARD INDUSTRIAL",
          "SHERWOOD",
          "SIGNAL HILL",
          "SILVER SPRINGS",
          "SILVERADO",
          "SKYVIEW RANCH",
          "SOMERSET",
          "SOUTH CALGARY",
          "SOUTHVIEW",
          "SOUTHWOOD",
          "SPRINGBANK HILL",
          "SPRUCE CLIFF",
          "ST. ANDREWS HEIGHTS",
          "STRATHCONA PARK",
          "SUNALTA",
          "SUNDANCE",
          "SUNNYSIDE",
          "SUNRIDGE",
          "TARADALE",
          "TEMPLE",
          "THORNCLIFFE",
          "TUSCANY",
          "TUXEDO PARK",
          "UNIVERSITY DISTRICT",
          "UNIVERSITY HEIGHTS",
          "UNIVERSITY OF CALGARY",
          "UPPER MOUNT ROYAL",
          "VALLEY RIDGE",
          "VARSITY",
          "VISTA HEIGHTS",
          "WALDEN",
          "WEST HILLHURST",
          "WEST SPRINGS",
          "WESTGATE",
          "WHITEHORN",
          "WILDWOOD",
          "WILLOW PARK",
          "WINDSOR PARK",
          "WINSTON HEIGHTS/MOUNTVIEW",
          "WOODBINE",
          "WOODLANDS",
          "YORKVILLE"
         ],
         "name": "",
         "parents": [
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          ""
         ],
         "type": "treemap",
         "values": [
          111.63,
          176.81,
          274.83,
          146.79,
          96.12,
          122.7,
          54.41,
          52.48,
          305.08,
          272.64,
          129.16,
          101.62,
          138.46,
          313.95,
          376.9,
          3731.71,
          63.58,
          217.85,
          89.58,
          198.71,
          297.73,
          43.27,
          191.3,
          136.54,
          111.88,
          273.82,
          218.53,
          144.28,
          74.66,
          41.41,
          113.48,
          172,
          111.31,
          88.82,
          37.21,
          84.41,
          475.46,
          71.21,
          142.29,
          49.99,
          77.57,
          160.5,
          47.73,
          79.9,
          220.25,
          57.62,
          44.51,
          319.34,
          63.91,
          28.53,
          111.19,
          91.62,
          52.32,
          72.46,
          52.1,
          69.56,
          204.13,
          791.89,
          480.35,
          214,
          751.66,
          268.97,
          50.86,
          132.56,
          200.11,
          136.9,
          266.14,
          37.66,
          33.81,
          170.32,
          183.23,
          4851.74,
          195.66,
          383.41,
          4520.33,
          70.39,
          40.12,
          127.11,
          117.84,
          191.32,
          228.23,
          3977.78,
          91.52,
          40.5,
          96.77,
          50.39,
          200.85,
          51.62,
          276.19,
          219.66,
          304.36,
          368.12,
          139.64,
          272.86,
          149.24,
          191.93,
          53.71,
          279.22,
          111.34,
          94.3,
          74.3,
          117.69,
          114.42,
          294.76,
          55.53,
          53.46,
          633.17,
          111.69,
          241.21,
          166.14,
          123.55,
          125,
          205.17,
          52.07,
          70.17,
          1408.39,
          89,
          70.87,
          287.52,
          82.72,
          312.07,
          245.71,
          46.63,
          44.24,
          161.02,
          93.26,
          73.46,
          73.81,
          149.25,
          89.87,
          48.89,
          154.72,
          308.1,
          56.56,
          79.01,
          176.64,
          2000,
          202.45,
          80.42,
          84.15,
          145.05,
          71.68,
          205.75,
          74.54,
          149.94,
          73.02,
          237.16,
          204.76,
          163.3,
          75.08,
          40.72,
          166.03,
          145.93,
          306.48,
          127.96,
          43.4,
          173.6,
          141.85,
          87.18,
          11360,
          79.76,
          55.72,
          197.64,
          147.13,
          62.35,
          495.59,
          271.83,
          152.28,
          135.99,
          1956.86,
          69,
          89.96,
          80.19,
          46.24,
          103.1,
          70.35,
          248.54,
          406.65,
          217.58,
          56.22,
          107.43,
          317.55,
          54.41,
          410.31,
          53.51,
          266.43,
          190363.64,
          79.42,
          152.77,
          161.24,
          40.23,
          297.78,
          255.32,
          142.42,
          59.99,
          163.03,
          50.31,
          137.49,
          240.08,
          58.45,
          187.68,
          53.91,
          93.69,
          135.29,
          117.02,
          208.15,
          229.49,
          267.4,
          60.91,
          97.13,
          1785.71
         ]
        }
       ],
       "layout": {
        "height": 1000,
        "legend": {
         "tracegroupgap": 0
        },
        "margin": {
         "t": 60
        },
        "template": {
         "data": {
          "bar": [
           {
            "error_x": {
             "color": "#2a3f5f"
            },
            "error_y": {
             "color": "#2a3f5f"
            },
            "marker": {
             "line": {
              "color": "#E5ECF6",
              "width": 0.5
             },
             "pattern": {
              "fillmode": "overlay",
              "size": 10,
              "solidity": 0.2
             }
            },
            "type": "bar"
           }
          ],
          "barpolar": [
           {
            "marker": {
             "line": {
              "color": "#E5ECF6",
              "width": 0.5
             },
             "pattern": {
              "fillmode": "overlay",
              "size": 10,
              "solidity": 0.2
             }
            },
            "type": "barpolar"
           }
          ],
          "carpet": [
           {
            "aaxis": {
             "endlinecolor": "#2a3f5f",
             "gridcolor": "white",
             "linecolor": "white",
             "minorgridcolor": "white",
             "startlinecolor": "#2a3f5f"
            },
            "baxis": {
             "endlinecolor": "#2a3f5f",
             "gridcolor": "white",
             "linecolor": "white",
             "minorgridcolor": "white",
             "startlinecolor": "#2a3f5f"
            },
            "type": "carpet"
           }
          ],
          "choropleth": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "type": "choropleth"
           }
          ],
          "contour": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "contour"
           }
          ],
          "contourcarpet": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "type": "contourcarpet"
           }
          ],
          "heatmap": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "heatmap"
           }
          ],
          "heatmapgl": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "heatmapgl"
           }
          ],
          "histogram": [
           {
            "marker": {
             "pattern": {
              "fillmode": "overlay",
              "size": 10,
              "solidity": 0.2
             }
            },
            "type": "histogram"
           }
          ],
          "histogram2d": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "histogram2d"
           }
          ],
          "histogram2dcontour": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "histogram2dcontour"
           }
          ],
          "mesh3d": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "type": "mesh3d"
           }
          ],
          "parcoords": [
           {
            "line": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "parcoords"
           }
          ],
          "pie": [
           {
            "automargin": true,
            "type": "pie"
           }
          ],
          "scatter": [
           {
            "fillpattern": {
             "fillmode": "overlay",
             "size": 10,
             "solidity": 0.2
            },
            "type": "scatter"
           }
          ],
          "scatter3d": [
           {
            "line": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatter3d"
           }
          ],
          "scattercarpet": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattercarpet"
           }
          ],
          "scattergeo": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattergeo"
           }
          ],
          "scattergl": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattergl"
           }
          ],
          "scattermapbox": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattermapbox"
           }
          ],
          "scatterpolar": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatterpolar"
           }
          ],
          "scatterpolargl": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatterpolargl"
           }
          ],
          "scatterternary": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatterternary"
           }
          ],
          "surface": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "surface"
           }
          ],
          "table": [
           {
            "cells": {
             "fill": {
              "color": "#EBF0F8"
             },
             "line": {
              "color": "white"
             }
            },
            "header": {
             "fill": {
              "color": "#C8D4E3"
             },
             "line": {
              "color": "white"
             }
            },
            "type": "table"
           }
          ]
         },
         "layout": {
          "annotationdefaults": {
           "arrowcolor": "#2a3f5f",
           "arrowhead": 0,
           "arrowwidth": 1
          },
          "autotypenumbers": "strict",
          "coloraxis": {
           "colorbar": {
            "outlinewidth": 0,
            "ticks": ""
           }
          },
          "colorscale": {
           "diverging": [
            [
             0,
             "#8e0152"
            ],
            [
             0.1,
             "#c51b7d"
            ],
            [
             0.2,
             "#de77ae"
            ],
            [
             0.3,
             "#f1b6da"
            ],
            [
             0.4,
             "#fde0ef"
            ],
            [
             0.5,
             "#f7f7f7"
            ],
            [
             0.6,
             "#e6f5d0"
            ],
            [
             0.7,
             "#b8e186"
            ],
            [
             0.8,
             "#7fbc41"
            ],
            [
             0.9,
             "#4d9221"
            ],
            [
             1,
             "#276419"
            ]
           ],
           "sequential": [
            [
             0,
             "#0d0887"
            ],
            [
             0.1111111111111111,
             "#46039f"
            ],
            [
             0.2222222222222222,
             "#7201a8"
            ],
            [
             0.3333333333333333,
             "#9c179e"
            ],
            [
             0.4444444444444444,
             "#bd3786"
            ],
            [
             0.5555555555555556,
             "#d8576b"
            ],
            [
             0.6666666666666666,
             "#ed7953"
            ],
            [
             0.7777777777777778,
             "#fb9f3a"
            ],
            [
             0.8888888888888888,
             "#fdca26"
            ],
            [
             1,
             "#f0f921"
            ]
           ],
           "sequentialminus": [
            [
             0,
             "#0d0887"
            ],
            [
             0.1111111111111111,
             "#46039f"
            ],
            [
             0.2222222222222222,
             "#7201a8"
            ],
            [
             0.3333333333333333,
             "#9c179e"
            ],
            [
             0.4444444444444444,
             "#bd3786"
            ],
            [
             0.5555555555555556,
             "#d8576b"
            ],
            [
             0.6666666666666666,
             "#ed7953"
            ],
            [
             0.7777777777777778,
             "#fb9f3a"
            ],
            [
             0.8888888888888888,
             "#fdca26"
            ],
            [
             1,
             "#f0f921"
            ]
           ]
          },
          "colorway": [
           "#636efa",
           "#EF553B",
           "#00cc96",
           "#ab63fa",
           "#FFA15A",
           "#19d3f3",
           "#FF6692",
           "#B6E880",
           "#FF97FF",
           "#FECB52"
          ],
          "font": {
           "color": "#2a3f5f"
          },
          "geo": {
           "bgcolor": "white",
           "lakecolor": "white",
           "landcolor": "#E5ECF6",
           "showlakes": true,
           "showland": true,
           "subunitcolor": "white"
          },
          "hoverlabel": {
           "align": "left"
          },
          "hovermode": "closest",
          "mapbox": {
           "style": "light"
          },
          "paper_bgcolor": "white",
          "plot_bgcolor": "#E5ECF6",
          "polar": {
           "angularaxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           },
           "bgcolor": "#E5ECF6",
           "radialaxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           }
          },
          "scene": {
           "xaxis": {
            "backgroundcolor": "#E5ECF6",
            "gridcolor": "white",
            "gridwidth": 2,
            "linecolor": "white",
            "showbackground": true,
            "ticks": "",
            "zerolinecolor": "white"
           },
           "yaxis": {
            "backgroundcolor": "#E5ECF6",
            "gridcolor": "white",
            "gridwidth": 2,
            "linecolor": "white",
            "showbackground": true,
            "ticks": "",
            "zerolinecolor": "white"
           },
           "zaxis": {
            "backgroundcolor": "#E5ECF6",
            "gridcolor": "white",
            "gridwidth": 2,
            "linecolor": "white",
            "showbackground": true,
            "ticks": "",
            "zerolinecolor": "white"
           }
          },
          "shapedefaults": {
           "line": {
            "color": "#2a3f5f"
           }
          },
          "ternary": {
           "aaxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           },
           "baxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           },
           "bgcolor": "#E5ECF6",
           "caxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           }
          },
          "title": {
           "x": 0.05
          },
          "xaxis": {
           "automargin": true,
           "gridcolor": "white",
           "linecolor": "white",
           "ticks": "",
           "title": {
            "standoff": 15
           },
           "zerolinecolor": "white",
           "zerolinewidth": 2
          },
          "yaxis": {
           "automargin": true,
           "gridcolor": "white",
           "linecolor": "white",
           "ticks": "",
           "title": {
            "standoff": 15
           },
           "zerolinecolor": "white",
           "zerolinewidth": 2
          }
         }
        },
        "title": {
         "text": "Crime Rates per Capita"
        },
        "width": 1500
       }
      }
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "import pandas as pd\n",
    "import plotly.express as px\n",
    "\n",
    "df = pd.read_csv('Data_2.csv')\n",
    "fig = px.treemap(df, path=['Community Name'], values='crime_rate_per_1000')\n",
    "fig.update_layout(title='Crime Rates per Capita', width=1500, height=1000)\n",
    "fig.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "As we can see here, 'Sunridge' is the community with most crimes per 1000 people, followed by 'Saddle Ridge Industrial'. This is an important information, because is different from the previous graphs form what we saw previously (showing that most crimes are located in the centre of the city)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Relationship Between Crimes and People"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "For this question, I want to know if there is a relationship between the number of people and number of crimes. For this I will display a scatterplot using 'Resident Count' and the sum of 'Crime count'."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/vnd.plotly.v1+json": {
       "config": {
        "plotlyServerURL": "https://plot.ly"
       },
       "data": [
        {
         "hovertemplate": "Crime Count=%{x}<br>Resident Count=%{y}<extra></extra>",
         "legendgroup": "",
         "marker": {
          "color": "#636efa",
          "symbol": "circle"
         },
         "mode": "markers",
         "name": "",
         "orientation": "v",
         "showlegend": false,
         "type": "scatter",
         "x": [
          19857,
          2094,
          53,
          284,
          153,
          27,
          556,
          895,
          499,
          227,
          1538,
          54,
          59,
          54,
          54,
          66,
          125,
          97,
          44,
          907,
          182,
          180,
          97,
          82,
          59,
          184,
          649,
          562,
          183,
          36,
          338,
          108,
          169,
          259,
          239,
          442,
          138,
          185,
          108,
          521,
          351,
          570,
          734,
          901,
          435,
          291,
          214,
          546,
          278,
          145,
          189,
          444,
          320,
          496,
          321,
          344,
          220,
          142,
          569,
          385,
          425,
          404,
          397,
          572,
          308,
          425,
          317,
          529,
          596,
          1030,
          416,
          262,
          125,
          300,
          1329,
          231,
          443,
          1019,
          405,
          1111,
          972,
          621,
          207,
          330,
          301,
          1060,
          1870,
          364,
          1098,
          1267,
          332,
          1127,
          223,
          1104,
          1409,
          1052,
          1322,
          502,
          1309,
          340,
          1299,
          264,
          1016,
          265,
          1433,
          1586,
          1109,
          505,
          281,
          420,
          450,
          520,
          427,
          1447,
          329,
          665,
          1223,
          583,
          455,
          884,
          364,
          1790,
          477,
          1210,
          1271,
          829,
          1996,
          1561,
          2114,
          2035,
          373,
          370,
          489,
          1019,
          671,
          2257,
          965,
          1422,
          1444,
          647,
          299,
          718,
          877,
          332,
          567,
          853,
          354,
          1475,
          2996,
          632,
          503,
          342,
          1416,
          601,
          1511,
          1280,
          696,
          6876,
          1417,
          540,
          994,
          2210,
          694,
          466,
          1274,
          514,
          521,
          559,
          2029,
          375,
          1146,
          2113,
          858,
          1860,
          1303,
          1952,
          580,
          1677,
          2429,
          1170,
          597,
          2029,
          508,
          1207,
          630,
          1626,
          547,
          524,
          1770,
          896,
          611,
          1205,
          698,
          1874,
          691,
          1760,
          783,
          924,
          1018,
          666,
          1283,
          1511,
          1685,
          727,
          1946,
          9471,
          1257
         ],
         "xaxis": "x",
         "y": [
          0,
          11,
          14,
          25,
          41,
          86,
          123,
          225,
          255,
          302,
          317,
          390,
          401,
          422,
          432,
          455,
          572,
          594,
          599,
          644,
          690,
          705,
          751,
          896,
          928,
          931,
          1025,
          1134,
          1254,
          1262,
          1270,
          1343,
          1477,
          1560,
          1594,
          1626,
          1640,
          1662,
          1690,
          1691,
          1754,
          1795,
          1805,
          1895,
          1906,
          1911,
          1916,
          2030,
          2036,
          2060,
          2128,
          2158,
          2249,
          2258,
          2263,
          2305,
          2359,
          2367,
          2370,
          2391,
          2471,
          2478,
          2566,
          2597,
          2617,
          2648,
          2709,
          2765,
          2785,
          2798,
          2921,
          3104,
          3116,
          3202,
          3239,
          3244,
          3342,
          3457,
          3569,
          3625,
          3635,
          3646,
          3660,
          3672,
          3767,
          3838,
          3893,
          3973,
          4024,
          4153,
          4202,
          4230,
          4280,
          4442,
          4515,
          4584,
          4598,
          4673,
          4688,
          4743,
          4744,
          4754,
          4962,
          5065,
          5256,
          5326,
          5328,
          5355,
          5585,
          5690,
          5801,
          5805,
          5848,
          5889,
          5904,
          5957,
          5961,
          6002,
          6094,
          6127,
          6228,
          6246,
          6420,
          6447,
          6496,
          6522,
          6558,
          6582,
          6620,
          6835,
          6855,
          6889,
          6900,
          6942,
          6981,
          6997,
          7049,
          7080,
          7267,
          7270,
          7382,
          7420,
          7442,
          7505,
          7607,
          7624,
          7655,
          7685,
          7814,
          7924,
          8067,
          8398,
          8523,
          8543,
          8554,
          8576,
          8679,
          8683,
          8788,
          8866,
          8940,
          9162,
          9244,
          9248,
          9368,
          9446,
          9736,
          9943,
          10022,
          10077,
          10293,
          10351,
          10372,
          10520,
          10619,
          10653,
          10758,
          10977,
          11150,
          11514,
          11566,
          11688,
          11706,
          11707,
          11784,
          12019,
          12641,
          12654,
          12874,
          12881,
          13103,
          13395,
          13406,
          13420,
          13823,
          14245,
          15395,
          17607,
          17667,
          17685,
          18283,
          19026,
          19884,
          21500,
          22321,
          25129,
          25710
         ],
         "yaxis": "y"
        },
        {
         "hovertemplate": "<b>OLS trendline</b><br>Resident Count = 0.63721 * Crime Count + 5439.43<br>R<sup>2</sup>=0.043025<br><br>Crime Count=%{x}<br>Resident Count=%{y} <b>(trend)</b><extra></extra>",
         "legendgroup": "",
         "marker": {
          "color": "#636efa",
          "symbol": "circle"
         },
         "mode": "lines",
         "name": "",
         "showlegend": false,
         "type": "scatter",
         "x": [
          27,
          36,
          44,
          53,
          54,
          54,
          54,
          59,
          59,
          66,
          82,
          97,
          97,
          108,
          108,
          125,
          125,
          138,
          142,
          145,
          153,
          169,
          180,
          182,
          183,
          184,
          185,
          189,
          207,
          214,
          220,
          223,
          227,
          231,
          239,
          259,
          262,
          264,
          265,
          278,
          281,
          284,
          291,
          299,
          300,
          301,
          308,
          317,
          320,
          321,
          329,
          330,
          332,
          332,
          338,
          340,
          342,
          344,
          351,
          354,
          364,
          364,
          370,
          373,
          375,
          385,
          397,
          404,
          405,
          416,
          420,
          425,
          425,
          427,
          435,
          442,
          443,
          444,
          450,
          455,
          466,
          477,
          489,
          496,
          499,
          502,
          503,
          505,
          508,
          514,
          520,
          521,
          521,
          524,
          529,
          540,
          546,
          547,
          556,
          559,
          562,
          567,
          569,
          570,
          572,
          580,
          583,
          596,
          597,
          601,
          611,
          621,
          630,
          632,
          647,
          649,
          665,
          666,
          671,
          691,
          694,
          696,
          698,
          718,
          727,
          734,
          783,
          829,
          853,
          858,
          877,
          884,
          895,
          896,
          901,
          907,
          924,
          965,
          972,
          994,
          1016,
          1018,
          1019,
          1019,
          1030,
          1052,
          1060,
          1098,
          1104,
          1109,
          1111,
          1127,
          1146,
          1170,
          1205,
          1207,
          1210,
          1223,
          1257,
          1267,
          1271,
          1274,
          1280,
          1283,
          1299,
          1303,
          1309,
          1322,
          1329,
          1409,
          1416,
          1417,
          1422,
          1433,
          1444,
          1447,
          1475,
          1511,
          1511,
          1538,
          1561,
          1586,
          1626,
          1677,
          1685,
          1760,
          1770,
          1790,
          1860,
          1870,
          1874,
          1946,
          1952,
          1996,
          2029,
          2029,
          2035,
          2094,
          2113,
          2114,
          2210,
          2257,
          2429,
          2996,
          6876,
          9471,
          19857
         ],
         "xaxis": "x",
         "y": [
          5456.636179167367,
          5462.371072426379,
          5467.468755323279,
          5473.203648582292,
          5473.840858944404,
          5473.840858944404,
          5473.840858944404,
          5477.026910754967,
          5477.026910754967,
          5481.487383289754,
          5491.682749083553,
          5501.240904515241,
          5501.240904515241,
          5508.250218498478,
          5508.250218498478,
          5519.08279465439,
          5519.08279465439,
          5527.366529361852,
          5529.915370810302,
          5531.82700189664,
          5536.92468479354,
          5547.120050587339,
          5554.129364570576,
          5555.403785294801,
          5556.040995656915,
          5556.678206019027,
          5557.31541638114,
          5559.864257829589,
          5571.334044347614,
          5575.794516882401,
          5579.617779055076,
          5581.529410141414,
          5584.078251589864,
          5586.627093038313,
          5591.724775935213,
          5604.468983177463,
          5606.3806142638,
          5607.655034988025,
          5608.292245350138,
          5616.5759800576,
          5618.487611143937,
          5620.399242230275,
          5624.859714765063,
          5629.957397661962,
          5630.594608024075,
          5631.231818386187,
          5635.692290920974,
          5641.427184179987,
          5643.338815266325,
          5643.976025628437,
          5649.073708525337,
          5649.710918887449,
          5650.985339611674,
          5650.985339611674,
          5654.808601784349,
          5656.083022508574,
          5657.357443232799,
          5658.631863957024,
          5663.092336491812,
          5665.003967578149,
          5671.376071199274,
          5671.376071199274,
          5675.199333371948,
          5677.110964458286,
          5678.385385182511,
          5684.757488803636,
          5692.4040131489855,
          5696.864485683773,
          5697.501696045885,
          5704.511010029123,
          5707.059851477573,
          5710.245903288135,
          5710.245903288135,
          5711.52032401236,
          5716.61800690926,
          5721.078479444047,
          5721.715689806159,
          5722.352900168272,
          5726.1761623409475,
          5729.36221415151,
          5736.371528134747,
          5743.380842117984,
          5751.027366463334,
          5755.487838998121,
          5757.399470084459,
          5759.311101170796,
          5759.9483115329085,
          5761.222732257133,
          5763.134363343471,
          5766.957625516146,
          5770.780887688821,
          5771.418098050934,
          5771.418098050934,
          5773.329729137271,
          5776.515780947833,
          5783.52509493107,
          5787.348357103745,
          5787.9855674658575,
          5793.7204607248705,
          5795.632091811208,
          5797.543722897545,
          5800.7297747081075,
          5802.004195432332,
          5802.641405794445,
          5803.91582651867,
          5809.01350941557,
          5810.925140501908,
          5819.20887520937,
          5819.846085571482,
          5822.394927019932,
          5828.767030641056,
          5835.139134262182,
          5840.874027521194,
          5842.148448245419,
          5851.706603677107,
          5852.9810244013315,
          5863.176390195131,
          5863.813600557243,
          5866.9996523678055,
          5879.743859610056,
          5881.655490696393,
          5882.929911420618,
          5884.204332144843,
          5896.948539387093,
          5902.683432646105,
          5907.143905180892,
          5938.367212924404,
          5967.678889581578,
          5982.971938272278,
          5986.15799008284,
          5998.264986962977,
          6002.725459497765,
          6009.734773481002,
          6010.371983843114,
          6013.558035653677,
          6017.381297826351,
          6028.213873982264,
          6054.339498828876,
          6058.799971363663,
          6072.818599330138,
          6086.837227296613,
          6088.1116480208375,
          6088.74885838295,
          6088.74885838295,
          6095.758172366187,
          6109.776800332662,
          6114.8744832295615,
          6139.088476989836,
          6142.9117391625105,
          6146.097790973074,
          6147.372211697299,
          6157.567577491098,
          6169.6745743712345,
          6184.967623061934,
          6207.269985735871,
          6208.544406460096,
          6210.456037546433,
          6218.739772253896,
          6240.404924565721,
          6246.777028186845,
          6249.325869635295,
          6251.237500721632,
          6255.060762894308,
          6256.972393980645,
          6267.167759774445,
          6269.7166012228945,
          6273.53986339557,
          6281.823598103032,
          6286.284070637819,
          6337.2608996068175,
          6341.721372141606,
          6342.358582503718,
          6345.54463431428,
          6352.553948297517,
          6359.563262280755,
          6361.474893367093,
          6379.316783506241,
          6402.2563565422915,
          6402.2563565422915,
          6419.461036319328,
          6434.116874647915,
          6450.0471337007275,
          6475.535548185227,
          6508.033276652963,
          6513.130959549863,
          6560.921736708299,
          6567.293840329424,
          6580.038047571674,
          6624.6427729195475,
          6631.014876540672,
          6633.563717989122,
          6679.442864061221,
          6683.266126233896,
          6711.303382166845,
          6732.331324116557,
          6732.331324116557,
          6736.154586289232,
          6773.749997653868,
          6785.856994534006,
          6786.494204896118,
          6847.666399658917,
          6877.615286678203,
          6987.21546896155,
          7348.5137442793275,
          9820.88994927576,
          11474.450838957653,
          18092.5176598579
         ],
         "yaxis": "y"
        }
       ],
       "layout": {
        "legend": {
         "tracegroupgap": 0
        },
        "margin": {
         "t": 60
        },
        "template": {
         "data": {
          "bar": [
           {
            "error_x": {
             "color": "#2a3f5f"
            },
            "error_y": {
             "color": "#2a3f5f"
            },
            "marker": {
             "line": {
              "color": "#E5ECF6",
              "width": 0.5
             },
             "pattern": {
              "fillmode": "overlay",
              "size": 10,
              "solidity": 0.2
             }
            },
            "type": "bar"
           }
          ],
          "barpolar": [
           {
            "marker": {
             "line": {
              "color": "#E5ECF6",
              "width": 0.5
             },
             "pattern": {
              "fillmode": "overlay",
              "size": 10,
              "solidity": 0.2
             }
            },
            "type": "barpolar"
           }
          ],
          "carpet": [
           {
            "aaxis": {
             "endlinecolor": "#2a3f5f",
             "gridcolor": "white",
             "linecolor": "white",
             "minorgridcolor": "white",
             "startlinecolor": "#2a3f5f"
            },
            "baxis": {
             "endlinecolor": "#2a3f5f",
             "gridcolor": "white",
             "linecolor": "white",
             "minorgridcolor": "white",
             "startlinecolor": "#2a3f5f"
            },
            "type": "carpet"
           }
          ],
          "choropleth": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "type": "choropleth"
           }
          ],
          "contour": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "contour"
           }
          ],
          "contourcarpet": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "type": "contourcarpet"
           }
          ],
          "heatmap": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "heatmap"
           }
          ],
          "heatmapgl": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "heatmapgl"
           }
          ],
          "histogram": [
           {
            "marker": {
             "pattern": {
              "fillmode": "overlay",
              "size": 10,
              "solidity": 0.2
             }
            },
            "type": "histogram"
           }
          ],
          "histogram2d": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "histogram2d"
           }
          ],
          "histogram2dcontour": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "histogram2dcontour"
           }
          ],
          "mesh3d": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "type": "mesh3d"
           }
          ],
          "parcoords": [
           {
            "line": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "parcoords"
           }
          ],
          "pie": [
           {
            "automargin": true,
            "type": "pie"
           }
          ],
          "scatter": [
           {
            "fillpattern": {
             "fillmode": "overlay",
             "size": 10,
             "solidity": 0.2
            },
            "type": "scatter"
           }
          ],
          "scatter3d": [
           {
            "line": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatter3d"
           }
          ],
          "scattercarpet": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattercarpet"
           }
          ],
          "scattergeo": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattergeo"
           }
          ],
          "scattergl": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattergl"
           }
          ],
          "scattermapbox": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattermapbox"
           }
          ],
          "scatterpolar": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatterpolar"
           }
          ],
          "scatterpolargl": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatterpolargl"
           }
          ],
          "scatterternary": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatterternary"
           }
          ],
          "surface": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "surface"
           }
          ],
          "table": [
           {
            "cells": {
             "fill": {
              "color": "#EBF0F8"
             },
             "line": {
              "color": "white"
             }
            },
            "header": {
             "fill": {
              "color": "#C8D4E3"
             },
             "line": {
              "color": "white"
             }
            },
            "type": "table"
           }
          ]
         },
         "layout": {
          "annotationdefaults": {
           "arrowcolor": "#2a3f5f",
           "arrowhead": 0,
           "arrowwidth": 1
          },
          "autotypenumbers": "strict",
          "coloraxis": {
           "colorbar": {
            "outlinewidth": 0,
            "ticks": ""
           }
          },
          "colorscale": {
           "diverging": [
            [
             0,
             "#8e0152"
            ],
            [
             0.1,
             "#c51b7d"
            ],
            [
             0.2,
             "#de77ae"
            ],
            [
             0.3,
             "#f1b6da"
            ],
            [
             0.4,
             "#fde0ef"
            ],
            [
             0.5,
             "#f7f7f7"
            ],
            [
             0.6,
             "#e6f5d0"
            ],
            [
             0.7,
             "#b8e186"
            ],
            [
             0.8,
             "#7fbc41"
            ],
            [
             0.9,
             "#4d9221"
            ],
            [
             1,
             "#276419"
            ]
           ],
           "sequential": [
            [
             0,
             "#0d0887"
            ],
            [
             0.1111111111111111,
             "#46039f"
            ],
            [
             0.2222222222222222,
             "#7201a8"
            ],
            [
             0.3333333333333333,
             "#9c179e"
            ],
            [
             0.4444444444444444,
             "#bd3786"
            ],
            [
             0.5555555555555556,
             "#d8576b"
            ],
            [
             0.6666666666666666,
             "#ed7953"
            ],
            [
             0.7777777777777778,
             "#fb9f3a"
            ],
            [
             0.8888888888888888,
             "#fdca26"
            ],
            [
             1,
             "#f0f921"
            ]
           ],
           "sequentialminus": [
            [
             0,
             "#0d0887"
            ],
            [
             0.1111111111111111,
             "#46039f"
            ],
            [
             0.2222222222222222,
             "#7201a8"
            ],
            [
             0.3333333333333333,
             "#9c179e"
            ],
            [
             0.4444444444444444,
             "#bd3786"
            ],
            [
             0.5555555555555556,
             "#d8576b"
            ],
            [
             0.6666666666666666,
             "#ed7953"
            ],
            [
             0.7777777777777778,
             "#fb9f3a"
            ],
            [
             0.8888888888888888,
             "#fdca26"
            ],
            [
             1,
             "#f0f921"
            ]
           ]
          },
          "colorway": [
           "#636efa",
           "#EF553B",
           "#00cc96",
           "#ab63fa",
           "#FFA15A",
           "#19d3f3",
           "#FF6692",
           "#B6E880",
           "#FF97FF",
           "#FECB52"
          ],
          "font": {
           "color": "#2a3f5f"
          },
          "geo": {
           "bgcolor": "white",
           "lakecolor": "white",
           "landcolor": "#E5ECF6",
           "showlakes": true,
           "showland": true,
           "subunitcolor": "white"
          },
          "hoverlabel": {
           "align": "left"
          },
          "hovermode": "closest",
          "mapbox": {
           "style": "light"
          },
          "paper_bgcolor": "white",
          "plot_bgcolor": "#E5ECF6",
          "polar": {
           "angularaxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           },
           "bgcolor": "#E5ECF6",
           "radialaxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           }
          },
          "scene": {
           "xaxis": {
            "backgroundcolor": "#E5ECF6",
            "gridcolor": "white",
            "gridwidth": 2,
            "linecolor": "white",
            "showbackground": true,
            "ticks": "",
            "zerolinecolor": "white"
           },
           "yaxis": {
            "backgroundcolor": "#E5ECF6",
            "gridcolor": "white",
            "gridwidth": 2,
            "linecolor": "white",
            "showbackground": true,
            "ticks": "",
            "zerolinecolor": "white"
           },
           "zaxis": {
            "backgroundcolor": "#E5ECF6",
            "gridcolor": "white",
            "gridwidth": 2,
            "linecolor": "white",
            "showbackground": true,
            "ticks": "",
            "zerolinecolor": "white"
           }
          },
          "shapedefaults": {
           "line": {
            "color": "#2a3f5f"
           }
          },
          "ternary": {
           "aaxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           },
           "baxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           },
           "bgcolor": "#E5ECF6",
           "caxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           }
          },
          "title": {
           "x": 0.05
          },
          "xaxis": {
           "automargin": true,
           "gridcolor": "white",
           "linecolor": "white",
           "ticks": "",
           "title": {
            "standoff": 15
           },
           "zerolinecolor": "white",
           "zerolinewidth": 2
          },
          "yaxis": {
           "automargin": true,
           "gridcolor": "white",
           "linecolor": "white",
           "ticks": "",
           "title": {
            "standoff": 15
           },
           "zerolinecolor": "white",
           "zerolinewidth": 2
          }
         }
        },
        "title": {
         "text": "Scatterplot con Línea de Tendencia"
        },
        "xaxis": {
         "anchor": "y",
         "domain": [
          0,
          1
         ],
         "title": {
          "text": "Crime Count"
         }
        },
        "yaxis": {
         "anchor": "x",
         "domain": [
          0,
          1
         ],
         "title": {
          "text": "Resident Count"
         }
        }
       }
      }
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "import plotly.express as px\n",
    "import pandas as pd\n",
    "\n",
    "df = df_clean.groupby('Resident Count')['Crime Count'].sum().reset_index()\n",
    "fig = px.scatter(df, x='Crime Count', y='Resident Count', trendline='ols')\n",
    "fig.update_layout(title='Scatterplot con Línea de Tendencia')\n",
    "fig.show()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "In this graph, we can see that there is a relation between the number of people and crimes, but the next step is to eliminate those outliers, so we can see a more clear graph."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Without Outliers"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/vnd.plotly.v1+json": {
       "config": {
        "plotlyServerURL": "https://plot.ly"
       },
       "data": [
        {
         "hovertemplate": "Crime Count=%{x}<br>Resident Count=%{y}<extra></extra>",
         "legendgroup": "",
         "line": {
          "color": "red"
         },
         "marker": {
          "color": "#636efa",
          "symbol": "circle"
         },
         "mode": "markers",
         "name": "",
         "orientation": "v",
         "showlegend": false,
         "type": "scatter",
         "x": [
          2094,
          53,
          284,
          153,
          27,
          556,
          895,
          499,
          227,
          1538,
          54,
          59,
          54,
          54,
          66,
          125,
          97,
          44,
          907,
          182,
          180,
          97,
          82,
          59,
          184,
          649,
          562,
          183,
          36,
          338,
          108,
          169,
          259,
          239,
          442,
          138,
          185,
          108,
          521,
          351,
          570,
          734,
          901,
          435,
          291,
          214,
          546,
          278,
          145,
          189,
          444,
          320,
          496,
          321,
          344,
          220,
          142,
          569,
          385,
          425,
          404,
          397,
          572,
          308,
          425,
          317,
          529,
          596,
          1030,
          416,
          262,
          125,
          300,
          1329,
          231,
          443,
          1019,
          405,
          1111,
          972,
          621,
          207,
          330,
          301,
          1060,
          1870,
          364,
          1098,
          1267,
          332,
          1127,
          223,
          1104,
          1409,
          1052,
          1322,
          502,
          1309,
          340,
          1299,
          264,
          1016,
          265,
          1433,
          1586,
          1109,
          505,
          281,
          420,
          450,
          520,
          427,
          1447,
          329,
          665,
          1223,
          583,
          455,
          884,
          364,
          1790,
          477,
          1210,
          1271,
          829,
          1996,
          1561,
          2114,
          2035,
          373,
          370,
          489,
          1019,
          671,
          2257,
          965,
          1422,
          1444,
          647,
          299,
          718,
          877,
          332,
          567,
          853,
          354,
          1475,
          2996,
          632,
          503,
          342,
          1416,
          601,
          1511,
          1280,
          696,
          1417,
          540,
          994,
          2210,
          694,
          466,
          1274,
          514,
          521,
          559,
          2029,
          375,
          1146,
          2113,
          858,
          1860,
          1303,
          1952,
          580,
          1677,
          2429,
          1170,
          597,
          2029,
          508,
          1207,
          630,
          1626,
          547,
          524,
          1770,
          896,
          611,
          1205,
          698,
          1874,
          691,
          1760,
          783,
          924,
          1018,
          666,
          1283,
          1511,
          1685,
          727,
          1946,
          1257
         ],
         "xaxis": "x",
         "y": [
          11,
          14,
          25,
          41,
          86,
          123,
          225,
          255,
          302,
          317,
          390,
          401,
          422,
          432,
          455,
          572,
          594,
          599,
          644,
          690,
          705,
          751,
          896,
          928,
          931,
          1025,
          1134,
          1254,
          1262,
          1270,
          1343,
          1477,
          1560,
          1594,
          1626,
          1640,
          1662,
          1690,
          1691,
          1754,
          1795,
          1805,
          1895,
          1906,
          1911,
          1916,
          2030,
          2036,
          2060,
          2128,
          2158,
          2249,
          2258,
          2263,
          2305,
          2359,
          2367,
          2370,
          2391,
          2471,
          2478,
          2566,
          2597,
          2617,
          2648,
          2709,
          2765,
          2785,
          2798,
          2921,
          3104,
          3116,
          3202,
          3239,
          3244,
          3342,
          3457,
          3569,
          3625,
          3635,
          3646,
          3660,
          3672,
          3767,
          3838,
          3893,
          3973,
          4024,
          4153,
          4202,
          4230,
          4280,
          4442,
          4515,
          4584,
          4598,
          4673,
          4688,
          4743,
          4744,
          4754,
          4962,
          5065,
          5256,
          5326,
          5328,
          5355,
          5585,
          5690,
          5801,
          5805,
          5848,
          5889,
          5904,
          5957,
          5961,
          6002,
          6094,
          6127,
          6228,
          6246,
          6420,
          6447,
          6496,
          6522,
          6558,
          6582,
          6620,
          6835,
          6855,
          6889,
          6900,
          6942,
          6981,
          6997,
          7049,
          7080,
          7267,
          7270,
          7382,
          7420,
          7442,
          7505,
          7607,
          7624,
          7655,
          7685,
          7814,
          7924,
          8067,
          8398,
          8523,
          8543,
          8554,
          8576,
          8679,
          8788,
          8866,
          8940,
          9162,
          9244,
          9248,
          9368,
          9446,
          9736,
          9943,
          10022,
          10077,
          10293,
          10351,
          10372,
          10520,
          10619,
          10653,
          10758,
          10977,
          11150,
          11514,
          11566,
          11688,
          11706,
          11707,
          11784,
          12019,
          12641,
          12654,
          12874,
          12881,
          13103,
          13395,
          13406,
          13420,
          13823,
          14245,
          15395,
          17607,
          17667,
          17685,
          18283,
          19026,
          19884,
          21500,
          22321,
          25710
         ],
         "yaxis": "y"
        },
        {
         "hovertemplate": "<b>OLS trendline</b><br>Resident Count = 3.90596 * Crime Count + 2944.27<br>R<sup>2</sup>=0.231065<br><br>Crime Count=%{x}<br>Resident Count=%{y} <b>(trend)</b><extra></extra>",
         "legendgroup": "",
         "line": {
          "color": "red"
         },
         "marker": {
          "color": "#636efa",
          "symbol": "circle"
         },
         "mode": "lines",
         "name": "",
         "showlegend": false,
         "type": "scatter",
         "x": [
          27,
          36,
          44,
          53,
          54,
          54,
          54,
          59,
          59,
          66,
          82,
          97,
          97,
          108,
          108,
          125,
          125,
          138,
          142,
          145,
          153,
          169,
          180,
          182,
          183,
          184,
          185,
          189,
          207,
          214,
          220,
          223,
          227,
          231,
          239,
          259,
          262,
          264,
          265,
          278,
          281,
          284,
          291,
          299,
          300,
          301,
          308,
          317,
          320,
          321,
          329,
          330,
          332,
          332,
          338,
          340,
          342,
          344,
          351,
          354,
          364,
          364,
          370,
          373,
          375,
          385,
          397,
          404,
          405,
          416,
          420,
          425,
          425,
          427,
          435,
          442,
          443,
          444,
          450,
          455,
          466,
          477,
          489,
          496,
          499,
          502,
          503,
          505,
          508,
          514,
          520,
          521,
          521,
          524,
          529,
          540,
          546,
          547,
          556,
          559,
          562,
          567,
          569,
          570,
          572,
          580,
          583,
          596,
          597,
          601,
          611,
          621,
          630,
          632,
          647,
          649,
          665,
          666,
          671,
          691,
          694,
          696,
          698,
          718,
          727,
          734,
          783,
          829,
          853,
          858,
          877,
          884,
          895,
          896,
          901,
          907,
          924,
          965,
          972,
          994,
          1016,
          1018,
          1019,
          1019,
          1030,
          1052,
          1060,
          1098,
          1104,
          1109,
          1111,
          1127,
          1146,
          1170,
          1205,
          1207,
          1210,
          1223,
          1257,
          1267,
          1271,
          1274,
          1280,
          1283,
          1299,
          1303,
          1309,
          1322,
          1329,
          1409,
          1416,
          1417,
          1422,
          1433,
          1444,
          1447,
          1475,
          1511,
          1511,
          1538,
          1561,
          1586,
          1626,
          1677,
          1685,
          1760,
          1770,
          1790,
          1860,
          1870,
          1874,
          1946,
          1952,
          1996,
          2029,
          2029,
          2035,
          2094,
          2113,
          2114,
          2210,
          2257,
          2429,
          2996
         ],
         "xaxis": "x",
         "y": [
          3049.7293181256355,
          3084.8829392784955,
          3116.130602525482,
          3151.284223678342,
          3155.1901815842157,
          3155.1901815842157,
          3155.1901815842157,
          3174.7199711135822,
          3174.7199711135822,
          3202.0616764546953,
          3264.557002948669,
          3323.1463715367686,
          3323.1463715367686,
          3366.111908501375,
          3366.111908501375,
          3432.513192901222,
          3432.513192901222,
          3483.2906456775754,
          3498.9144773010685,
          3510.6323510186885,
          3541.880014265675,
          3604.375340759648,
          3647.340877724255,
          3655.152793536002,
          3659.058751441875,
          3662.9647093477483,
          3666.870667253622,
          3682.494498877115,
          3752.801741182835,
          3780.143446523948,
          3803.579193959188,
          3815.297067676808,
          3830.9208993003012,
          3846.5447309237948,
          3877.7923941707813,
          3955.911552288248,
          3967.629426005868,
          3975.4413418176146,
          3979.347299723488,
          4030.124752499841,
          4041.842626217461,
          4053.560499935081,
          4080.9022052761948,
          4112.149868523181,
          4116.055826429054,
          4119.961784334928,
          4147.303489676041,
          4182.457110828901,
          4194.174984546521,
          4198.080942452394,
          4229.328605699381,
          4233.234563605254,
          4241.046479417001,
          4241.046479417001,
          4264.482226852241,
          4272.294142663988,
          4280.106058475734,
          4287.917974287481,
          4315.259679628594,
          4326.977553346214,
          4366.037132404947,
          4366.037132404947,
          4389.4728798401875,
          4401.1907535578075,
          4409.0026693695545,
          4448.062248428288,
          4494.933743298768,
          4522.275448639881,
          4526.181406545754,
          4569.146943510361,
          4584.770775133854,
          4604.300564663221,
          4604.300564663221,
          4612.112480474967,
          4643.360143721954,
          4670.701849063067,
          4674.607806968941,
          4678.513764874814,
          4701.949512310054,
          4721.47930183942,
          4764.444838804027,
          4807.4103757686335,
          4854.281870639114,
          4881.6235759802275,
          4893.341449697847,
          4905.059323415467,
          4908.96528132134,
          4916.777197133087,
          4928.495070850707,
          4951.930818285947,
          4975.366565721187,
          4979.27252362706,
          4979.27252362706,
          4990.99039734468,
          5010.520186874047,
          5053.485723838654,
          5076.921471273894,
          5080.827429179766,
          5115.981050332626,
          5127.698924050246,
          5139.416797767866,
          5158.946587297233,
          5166.75850310898,
          5170.664461014853,
          5178.4763768266,
          5209.724040073586,
          5221.441913791206,
          5272.21936656756,
          5276.125324473433,
          5291.7491560969265,
          5330.80873515566,
          5369.868314214393,
          5405.021935367253,
          5412.833851179,
          5471.423219767099,
          5479.235135578847,
          5541.730462072819,
          5545.636419978693,
          5565.166209508059,
          5643.285367625526,
          5655.003241343146,
          5662.815157154893,
          5670.627072966639,
          5748.746231084106,
          5783.899852236966,
          5811.24155757808,
          6002.633494965872,
          6182.3075586360455,
          6276.050548377005,
          6295.580337906373,
          6369.793538117965,
          6397.135243459079,
          6440.100780423685,
          6444.006738329559,
          6463.536527858925,
          6486.972275294165,
          6553.373559694011,
          6713.517833834818,
          6740.859539175932,
          6826.790613105145,
          6912.721687034358,
          6920.533602846104,
          6924.439560751978,
          6924.439560751978,
          6967.405097716584,
          7053.3361716457985,
          7084.583834892785,
          7233.010235315971,
          7256.445982751211,
          7275.975772280577,
          7283.787688092325,
          7346.283014586297,
          7420.496214797891,
          7514.2392045388515,
          7650.947731244418,
          7658.759647056164,
          7670.477520773784,
          7721.254973550138,
          7854.05754234983,
          7893.117121408564,
          7908.7409530320565,
          7920.4588267496765,
          7943.894574184917,
          7955.612447902537,
          8018.107774396511,
          8033.731606020003,
          8057.167353455243,
          8107.944806231597,
          8135.286511572711,
          8447.763144042576,
          8475.10484938369,
          8479.010807289564,
          8498.54059681893,
          8541.506133783536,
          8584.471670748144,
          8596.189544465762,
          8705.556365830216,
          8846.170850441657,
          8846.170850441657,
          8951.631713900237,
          9041.468745735323,
          9139.117693382155,
          9295.35600961709,
          9494.559862816628,
          9525.807526063616,
          9818.754369004115,
          9857.813948062849,
          9935.933106180315,
          10209.350159591448,
          10248.409738650182,
          10264.033570273674,
          10545.262539496554,
          10568.698286931794,
          10740.56043479022,
          10869.457045684041,
          10869.457045684041,
          10892.892793119281,
          11123.344309565808,
          11197.5575097774,
          11201.463467683274,
          11576.435426647113,
          11760.01544822316,
          12431.840208033373,
          14646.51834066355
         ],
         "yaxis": "y"
        }
       ],
       "layout": {
        "height": 1000,
        "legend": {
         "tracegroupgap": 0
        },
        "margin": {
         "t": 60
        },
        "template": {
         "data": {
          "bar": [
           {
            "error_x": {
             "color": "#2a3f5f"
            },
            "error_y": {
             "color": "#2a3f5f"
            },
            "marker": {
             "line": {
              "color": "#E5ECF6",
              "width": 0.5
             },
             "pattern": {
              "fillmode": "overlay",
              "size": 10,
              "solidity": 0.2
             }
            },
            "type": "bar"
           }
          ],
          "barpolar": [
           {
            "marker": {
             "line": {
              "color": "#E5ECF6",
              "width": 0.5
             },
             "pattern": {
              "fillmode": "overlay",
              "size": 10,
              "solidity": 0.2
             }
            },
            "type": "barpolar"
           }
          ],
          "carpet": [
           {
            "aaxis": {
             "endlinecolor": "#2a3f5f",
             "gridcolor": "white",
             "linecolor": "white",
             "minorgridcolor": "white",
             "startlinecolor": "#2a3f5f"
            },
            "baxis": {
             "endlinecolor": "#2a3f5f",
             "gridcolor": "white",
             "linecolor": "white",
             "minorgridcolor": "white",
             "startlinecolor": "#2a3f5f"
            },
            "type": "carpet"
           }
          ],
          "choropleth": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "type": "choropleth"
           }
          ],
          "contour": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "contour"
           }
          ],
          "contourcarpet": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "type": "contourcarpet"
           }
          ],
          "heatmap": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "heatmap"
           }
          ],
          "heatmapgl": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "heatmapgl"
           }
          ],
          "histogram": [
           {
            "marker": {
             "pattern": {
              "fillmode": "overlay",
              "size": 10,
              "solidity": 0.2
             }
            },
            "type": "histogram"
           }
          ],
          "histogram2d": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "histogram2d"
           }
          ],
          "histogram2dcontour": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "histogram2dcontour"
           }
          ],
          "mesh3d": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "type": "mesh3d"
           }
          ],
          "parcoords": [
           {
            "line": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "parcoords"
           }
          ],
          "pie": [
           {
            "automargin": true,
            "type": "pie"
           }
          ],
          "scatter": [
           {
            "fillpattern": {
             "fillmode": "overlay",
             "size": 10,
             "solidity": 0.2
            },
            "type": "scatter"
           }
          ],
          "scatter3d": [
           {
            "line": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatter3d"
           }
          ],
          "scattercarpet": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattercarpet"
           }
          ],
          "scattergeo": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattergeo"
           }
          ],
          "scattergl": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattergl"
           }
          ],
          "scattermapbox": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattermapbox"
           }
          ],
          "scatterpolar": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatterpolar"
           }
          ],
          "scatterpolargl": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatterpolargl"
           }
          ],
          "scatterternary": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatterternary"
           }
          ],
          "surface": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "surface"
           }
          ],
          "table": [
           {
            "cells": {
             "fill": {
              "color": "#EBF0F8"
             },
             "line": {
              "color": "white"
             }
            },
            "header": {
             "fill": {
              "color": "#C8D4E3"
             },
             "line": {
              "color": "white"
             }
            },
            "type": "table"
           }
          ]
         },
         "layout": {
          "annotationdefaults": {
           "arrowcolor": "#2a3f5f",
           "arrowhead": 0,
           "arrowwidth": 1
          },
          "autotypenumbers": "strict",
          "coloraxis": {
           "colorbar": {
            "outlinewidth": 0,
            "ticks": ""
           }
          },
          "colorscale": {
           "diverging": [
            [
             0,
             "#8e0152"
            ],
            [
             0.1,
             "#c51b7d"
            ],
            [
             0.2,
             "#de77ae"
            ],
            [
             0.3,
             "#f1b6da"
            ],
            [
             0.4,
             "#fde0ef"
            ],
            [
             0.5,
             "#f7f7f7"
            ],
            [
             0.6,
             "#e6f5d0"
            ],
            [
             0.7,
             "#b8e186"
            ],
            [
             0.8,
             "#7fbc41"
            ],
            [
             0.9,
             "#4d9221"
            ],
            [
             1,
             "#276419"
            ]
           ],
           "sequential": [
            [
             0,
             "#0d0887"
            ],
            [
             0.1111111111111111,
             "#46039f"
            ],
            [
             0.2222222222222222,
             "#7201a8"
            ],
            [
             0.3333333333333333,
             "#9c179e"
            ],
            [
             0.4444444444444444,
             "#bd3786"
            ],
            [
             0.5555555555555556,
             "#d8576b"
            ],
            [
             0.6666666666666666,
             "#ed7953"
            ],
            [
             0.7777777777777778,
             "#fb9f3a"
            ],
            [
             0.8888888888888888,
             "#fdca26"
            ],
            [
             1,
             "#f0f921"
            ]
           ],
           "sequentialminus": [
            [
             0,
             "#0d0887"
            ],
            [
             0.1111111111111111,
             "#46039f"
            ],
            [
             0.2222222222222222,
             "#7201a8"
            ],
            [
             0.3333333333333333,
             "#9c179e"
            ],
            [
             0.4444444444444444,
             "#bd3786"
            ],
            [
             0.5555555555555556,
             "#d8576b"
            ],
            [
             0.6666666666666666,
             "#ed7953"
            ],
            [
             0.7777777777777778,
             "#fb9f3a"
            ],
            [
             0.8888888888888888,
             "#fdca26"
            ],
            [
             1,
             "#f0f921"
            ]
           ]
          },
          "colorway": [
           "#636efa",
           "#EF553B",
           "#00cc96",
           "#ab63fa",
           "#FFA15A",
           "#19d3f3",
           "#FF6692",
           "#B6E880",
           "#FF97FF",
           "#FECB52"
          ],
          "font": {
           "color": "#2a3f5f"
          },
          "geo": {
           "bgcolor": "white",
           "lakecolor": "white",
           "landcolor": "#E5ECF6",
           "showlakes": true,
           "showland": true,
           "subunitcolor": "white"
          },
          "hoverlabel": {
           "align": "left"
          },
          "hovermode": "closest",
          "mapbox": {
           "style": "light"
          },
          "paper_bgcolor": "white",
          "plot_bgcolor": "#E5ECF6",
          "polar": {
           "angularaxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           },
           "bgcolor": "#E5ECF6",
           "radialaxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           }
          },
          "scene": {
           "xaxis": {
            "backgroundcolor": "#E5ECF6",
            "gridcolor": "white",
            "gridwidth": 2,
            "linecolor": "white",
            "showbackground": true,
            "ticks": "",
            "zerolinecolor": "white"
           },
           "yaxis": {
            "backgroundcolor": "#E5ECF6",
            "gridcolor": "white",
            "gridwidth": 2,
            "linecolor": "white",
            "showbackground": true,
            "ticks": "",
            "zerolinecolor": "white"
           },
           "zaxis": {
            "backgroundcolor": "#E5ECF6",
            "gridcolor": "white",
            "gridwidth": 2,
            "linecolor": "white",
            "showbackground": true,
            "ticks": "",
            "zerolinecolor": "white"
           }
          },
          "shapedefaults": {
           "line": {
            "color": "#2a3f5f"
           }
          },
          "ternary": {
           "aaxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           },
           "baxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           },
           "bgcolor": "#E5ECF6",
           "caxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           }
          },
          "title": {
           "x": 0.05
          },
          "xaxis": {
           "automargin": true,
           "gridcolor": "white",
           "linecolor": "white",
           "ticks": "",
           "title": {
            "standoff": 15
           },
           "zerolinecolor": "white",
           "zerolinewidth": 2
          },
          "yaxis": {
           "automargin": true,
           "gridcolor": "white",
           "linecolor": "white",
           "ticks": "",
           "title": {
            "standoff": 15
           },
           "zerolinecolor": "white",
           "zerolinewidth": 2
          }
         }
        },
        "title": {
         "text": "Scatterplot con Línea de Tendencia"
        },
        "width": 1700,
        "xaxis": {
         "anchor": "y",
         "domain": [
          0,
          1
         ],
         "title": {
          "text": "Crime Count"
         }
        },
        "yaxis": {
         "anchor": "x",
         "domain": [
          0,
          1
         ],
         "title": {
          "text": "Resident Count"
         }
        }
       }
      }
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Correlation: 0.48\n"
     ]
    }
   ],
   "source": [
    "import plotly.express as px\n",
    "import pandas as pd\n",
    "\n",
    "df = df[df['Crime Count'] < 5000].groupby('Resident Count')['Crime Count'].sum().reset_index()\n",
    "df = df[df['Crime Count'] < 3000]\n",
    "\n",
    "fig = px.scatter(df, x='Crime Count', y='Resident Count', trendline='ols')\n",
    "fig.update_traces(line=dict(color='red'))\n",
    "fig.update_layout(title='Scatterplot con Línea de Tendencia', height=1000, width=1700)\n",
    "fig.show()\n",
    "\n",
    "print(\"Correlation: \" + str(round(df['Crime Count'].corr(df['Resident Count']),2)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [],
   "source": [
    "# df.to_csv('Data_3.csv')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Now we can see that there is a possitive correlation between the number of residents and crime. I also calculated the correlation between these two variables, showing that the correlation is 0.48, confirming what the graph shows us"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "base",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
