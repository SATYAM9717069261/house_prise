import sys
from PySide2 import QtCore
import secrets
from PySide2.QtUiTools import QUiLoader
from PySide2.QtWidgets import QApplication,QStackedWidget,QMessageBox
from PySide2.QtCore import QFile, QCoreApplication, QIODevice
from Setup import Setup_urls
import os
import csv
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost




class Main_window(Setup_urls):
    def __init__(self,url):
        super(Main_window,self).__init__(url)

    def btn_click(self):
        self.window.cal.clicked.connect(self.pridict) # quizz add in main frame
        self.window.clear.clicked.connect(self.clean) # fillup add in main Frame

    
    def pridict(self):
        self.input_data_formated()
        #self.prediction()
        chk=self.check_inp()
        if(chk==True):
            self.prediction()
        else:
            #popup
            print("Not ")

            QMessageBox.warning(self.window,"Warning","Fill all Field")

           


    def check_inp(self):
        if(self.window.inp_txt_4.text() == "" and self.window.inp_txt_3.text()=="" and self.window.inp_txt_18.text()=="" and self.window.inp_txt_19.text()=="" and self.window.inp_txt_25.text()=="" and self.window.inp_txt_34.text()=="" and self.window.inp_txt_35.text()=="" and self.window.inp_txt_36.text()=="" and self.window.inp_txt_41.text()=="" and self.window.inp_txt_42.text()=="" and self.window.inp_txt_43.text()=="" and self.window.inp_txt_44.text()=="" and self.window.inp_txt_45.text()=="" and self.window.inp_txt_46.text()=="" and self.window.inp_txt_47.text()=="" and self.window.inp_txt_48.text()=="" and self.window.inp_txt_49.text()=="" and self.window.inp_txt_50.text()=="" and self.window.inp_txt_52.text()=="" and self.window.inp_txt_54.text()=="" and self.window.inp_txt_58.text()=="" and self.window.inp_txt_59.text()=="" and self.window.inp_txt_63.text()=="" ):
            return False
        else:
            return True
    
    def clean(self):
        self.window.inp_txt_4.setText("")
        self.window.inp_txt_3.setText("")
        self.window.inp_txt_18.setText("")
        self.window.inp_txt_19.setText("")
        self.window.inp_txt_25.setText("")
        self.window.inp_txt_34.setText("")
        self.window.inp_txt_35.setText("")
        self.window.inp_txt_36.setText("")
        self.window.inp_txt_41.setText("")
        self.window.inp_txt_42.setText("")
        self.window.inp_txt_43.setText("")
        self.window.inp_txt_44.setText("")
        self.window.inp_txt_45.setText("")
        self.window.inp_txt_46.setText("")
        self.window.inp_txt_47.setText("")
        self.window.inp_txt_48.setText("")
        self.window.inp_txt_49.setText("")
        self.window.inp_txt_50.setText("")
        self.window.inp_txt_52.setText("")
        self.window.inp_txt_54.setText("")
        self.window.inp_txt_58.setText("")
        self.window.inp_txt_59.setText("")
        self.window.inp_txt_63.setText("")
        print("all")

    def chk_inp(self,inp):
        if(inp.isdigit()):
            return inp
        else:
            # show Alert
            #inp.str=""9852101508
            return "0"

    def input_data_formated(self):
        
        inp1=(self.window.inp_drop_1.currentText()).split(" ")[0]
        inp2=(self.window.inp_drop_2.currentText()).split(" ")[0]
        inp3=self.chk_inp( self.window.inp_txt_3.text())
        inp4=self.chk_inp( self.window.inp_txt_4.text())
        inp5=(self.window.inp_drop_5.currentText()).split(" ")[0]
        inp6=(self.window.inp_drop_6.currentText()).split(" ")[0]
        inp7=(self.window.inp_drop_7.currentText()).split(" ")[0]
        inp8=(self.window.inp_drop_8.currentText()).split(" ")[0]
        inp9=(self.window.inp_drop_9.currentText()).split(" ")[0]
        inp10=(self.window.inp_drop_10.currentText()).split(" ")[0]
        inp11=(self.window.inp_drop_11.currentText()).split(" ")[0]
        inp12=(self.window.inp_drop_12.currentText()).split(" ")[0]
        inp13=(self.window.inp_drop_13.currentText()).split(" ")[0]
        inp14=(self.window.inp_drop_14.currentText()).split(" ")[0]
        inp15=(self.window.inp_drop_15.currentText()).split(" ")[0]
        inp16=(self.window.inp_drop_16.currentText()).split(" ")[0]
        inp17=(self.window.inp_drop_17.currentText()).split(" ")[0]
        inp18=self.chk_inp( self.window.inp_txt_18.text())
        inp19=self.chk_inp( self.window.inp_txt_19.text())
        inp20=(self.window.inp_drop_20.currentText()).split(" ")[0]
        inp21=(self.window.inp_drop_21.currentText()).split(" ")[0]
        inp22=(self.window.inp_drop_22.currentText()).split(" ")[0]
        inp23=(self.window.inp_drop_23.currentText()).split(" ")[0]
        inp24=(self.window.inp_drop_24.currentText()).split(" ")[0]
        inp25=self.chk_inp( self.window.inp_txt_25.text())

        inp26=(self.window.inp_drop_26.currentText()).split(" ")[0]
        inp27=(self.window.inp_drop_27.currentText()).split(" ")[0]
        inp28=(self.window.inp_drop_28.currentText()).split(" ")[0]
        inp29=(self.window.inp_drop_29.currentText()).split(" ")[0]
        inp30=(self.window.inp_drop_30.currentText()).split(" ")[0]
        inp31=(self.window.inp_drop_31.currentText()).split(" ")[0]
        inp32=(self.window.inp_drop_32.currentText()).split(" ")[0]
        inp33=(self.window.inp_drop_33.currentText()).split(" ")[0]

        inp34=self.chk_inp( self.window.inp_txt_34.text())
        inp35=self.chk_inp( self.window.inp_txt_35.text())
        inp36=self.chk_inp( self.window.inp_txt_36.text())

        inp37=(self.window.inp_drop_37.currentText()).split(" ")[0]
        inp38=(self.window.inp_drop_38.currentText()).split(" ")[0]
        inp39=(self.window.inp_drop_39.currentText()).split(" ")[0]
        inp40=(self.window.inp_drop_40.currentText()).split(" ")[0]

        inp41=self.chk_inp( self.window.inp_txt_41.text())
        inp42=self.chk_inp( self.window.inp_txt_42.text())
        inp43=self.chk_inp( self.window.inp_txt_43.text())
        inp44=self.chk_inp( self.window.inp_txt_44.text())
        inp45=self.chk_inp( self.window.inp_txt_45.text())
        inp46=self.chk_inp( self.window.inp_txt_46.text())
        inp47=self.chk_inp( self.window.inp_txt_47.text())
        inp48=self.chk_inp( self.window.inp_txt_48.text())
        inp49=self.chk_inp( self.window.inp_txt_49.text())
        inp50=self.chk_inp( self.window.inp_txt_50.text())

        inp51=(self.window.inp_drop_51.currentText()).split(" ")[0]
        inp52=self.chk_inp( self.window.inp_txt_52.text())

        inp53=(self.window.inp_drop_53.currentText()).split(" ")[0]
        inp54=self.chk_inp( self.window.inp_txt_54.text())
        inp55=(self.window.inp_drop_55.currentText()).split(" ")[0]

        inp56=(self.window.inp_drop_56.currentText()).split(" ")[0]
        inp57=(self.window.inp_drop_57.currentText()).split(" ")[0]
        inp58=self.chk_inp( self.window.inp_txt_58.text())
        inp59=self.chk_inp( self.window.inp_txt_59.text())
        inp60=(self.window.inp_drop_60.currentText()).split(" ")[0]
        inp61=(self.window.inp_drop_61.currentText()).split(" ")[0]
        inp62=(self.window.inp_drop_62.currentText()).split(" ")[0]
        inp63=self.chk_inp( self.window.inp_txt_63.text())

        inp64='0'
        inp65='0'
        inp66='0'
        inp67="120"
        inp68='0'
        inp69='0'
        inp70='6'
        inp71="2010"
        inp72="WD"
        inp73="Normal"
        inp74="LwQ"
        self.data=[
                inp1,inp2,inp3,inp4,inp5,inp6,inp7,inp8,inp9,inp10,
                inp11,inp12,inp13,inp14,inp15,inp16,inp17,inp18,inp19,inp20,
                inp21,inp22,inp23,inp24,inp25,inp26,inp27,inp28,inp29,inp30,
                inp31,inp32,inp33,inp74,inp34,inp35,inp36,inp37,inp38,inp39,inp40,
                inp41,inp42,inp43,inp44,inp45,inp46,inp47,inp48,inp49,inp50,
                inp51,inp52,inp53,inp54,inp55,inp56,inp57,inp58,inp59,inp60,
                inp61,inp62,inp63,inp64,inp65,inp66,inp67,inp68,inp69,inp70,
                inp71,inp72,inp73
                ]


        #print(inp3)
        
    
    def prediction(self):
        df=pd.read_csv('train.csv')
        df['MSZoning'].value_counts()
        ## Fill Missing Values
        df['LotFrontage']=df['LotFrontage'].fillna(df['LotFrontage'].mean())
        df.drop(['Alley'],axis=1,inplace=True)
        df['BsmtCond']=df['BsmtCond'].fillna(df['BsmtCond'].mode()[0])
        df['BsmtQual']=df['BsmtQual'].fillna(df['BsmtQual'].mode()[0])
        df['FireplaceQu']=df['FireplaceQu'].fillna(df['FireplaceQu'].mode()[0])
        df['GarageType']=df['GarageType'].fillna(df['GarageType'].mode()[0])
        df.drop(['GarageYrBlt'],axis=1,inplace=True)
        df['GarageFinish']=df['GarageFinish'].fillna(df['GarageFinish'].mode()[0])
        df['GarageQual']=df['GarageQual'].fillna(df['GarageQual'].mode()[0])
        df['GarageCond']=df['GarageCond'].fillna(df['GarageCond'].mode()[0])
        df.drop(['PoolQC','Fence','MiscFeature'],axis=1,inplace=True)
        df.drop(['Id'],axis=1,inplace=True)
        df.isnull().sum()
        df['MasVnrType']=df['MasVnrType'].fillna(df['MasVnrType'].mode()[0])
        df['MasVnrArea']=df['MasVnrArea'].fillna(df['MasVnrArea'].mode()[0])
        df['BsmtExposure']=df['BsmtExposure'].fillna(df['BsmtExposure'].mode()[0])
        df['BsmtFinType2']=df['BsmtFinType2'].fillna(df['BsmtFinType2'].mode()[0])
        df.dropna(inplace=True)
        columns=['MSZoning','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood',
         'Condition2','BldgType','Condition1','HouseStyle','SaleType',
        'SaleCondition','ExterCond',
         'ExterQual','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',
        'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','Heating','HeatingQC',
         'CentralAir',
         'Electrical','KitchenQual','Functional',
         'FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive']

        def category_onehot_multcols(multcolumns):
            df_final=final_df
            i=0
            for fields in multcolumns:
                #print(fields)
                df1=pd.get_dummies(final_df[fields],drop_first=True)
                final_df.drop([fields],axis=1,inplace=True)
                
                if i==0:
                    df_final=df1.copy()
                else:
                    df_final=pd.concat([df_final,df1],axis=1)
                i=i+1
            df_final=pd.concat([final_df,df_final],axis=1)
            return df_final
        main_df=df.copy()

        test_df=pd.read_csv('formulatedtest.csv')
        
        final_df=pd.concat([df,test_df],axis=0)

        final_df['SalePrice']

        final_df=category_onehot_multcols(columns)

        final_df =final_df.loc[:,~final_df.columns.duplicated()]
        df_Train=final_df.iloc[:1422,:]
        df_Test=final_df.iloc[1422:,:]


        df_Test.drop(['SalePrice'],axis=1,inplace=True)

        X_train=df_Train.drop(['SalePrice'],axis=1)
        y_train=df_Train['SalePrice']
        classifier=xgboost.XGBRegressor()
        classifier.fit(X_train,y_train)
        y_pred=classifier.predict(df_Test)
        #print(secrets.choice(y_pred))
        self.window.out_price.setText(str(secrets.choice(y_pred)))

QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_ShareOpenGLContexts)
app = QApplication(sys.argv)
main=Main_window("../UI/cal_window.ui")
main.btn_click()
main.window.show()
sys.exit(app.exec_())


