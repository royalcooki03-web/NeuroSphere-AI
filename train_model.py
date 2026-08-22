from pathlib import Path
import pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
BASE=Path(__file__).parent
df=pd.read_csv(BASE/"data/student_success_dataset.csv")
features=["attendance","study_hours","previous_score","assignments","sleep_hours","screen_time","engagement"]
Xtr,Xte,ytr,yte=train_test_split(df[features],df.risk_level,test_size=.2,random_state=42,stratify=df.risk_level)
m=RandomForestClassifier(n_estimators=220,max_depth=9,min_samples_leaf=3,class_weight="balanced",random_state=42)
m.fit(Xtr,ytr)
print("Accuracy:",round(accuracy_score(yte,m.predict(Xte)),4))
joblib.dump({"model":m,"features":features},BASE/"models/risk_model.joblib")
