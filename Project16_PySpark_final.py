import os, sys, re, types, importlib.util
import pandas as pd
import numpy as np
from functools import reduce
_parse = lambda s: tuple(int(re.match(r"\d*", p).group() or 0) for p in re.split(r"[.\-+]", str(s)))
_ver = lambda x: list(x.version) if hasattr(x, "version") else list(_parse(x))
_LV = type("LooseVersion", (object,), {"__init__": lambda self, v: setattr(self, "version", list(_parse(v))), "__lt__": lambda a, b: _ver(a) < _ver(b), "__le__": lambda a, b: _ver(a) <= _ver(b), "__gt__": lambda a, b: _ver(a) > _ver(b), "__ge__": lambda a, b: _ver(a) >= _ver(b), "__eq__": lambda a, b: _ver(a) == _ver(b), "__ne__": lambda a, b: _ver(a) != _ver(b), "__hash__": lambda self: hash(tuple(self.version)), "__repr__": lambda self: "LooseVersion(%r)" % ".".join(map(str, self.version))})
_dist = types.ModuleType("distutils")
_distver = types.ModuleType("distutils.version")
_distver.LooseVersion = _LV
_dist.version = _distver
sys.modules.update({} if importlib.util.find_spec("distutils") else {"distutils": _dist, "distutils.version": _distver})
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.functions import vector_to_array
print("PySpark ML imports OK")
spark = SparkSession.builder.getOrCreate()
spark.sparkContext.setLogLevel("ERROR")
data_path = "C:/Users/prane/Downloads/hotel_bookings.csv"
out_dir = "C:/Users/prane/Downloads/project16_outputs"
os.makedirs(out_dir, exist_ok=True)

data = spark.read.csv(data_path, header=True, inferSchema=True, nullValue="NULL")
data.printSchema()
raw_rows = data.count()
print("rows:", raw_rows, "| columns:", len(data.columns))
data.select("hotel", "is_canceled", "lead_time", "customer_type", "deposit_type", "market_segment", "adr").show(5)

data.select([F.sum(F.col(c).isNull().cast("int")).alias(c) for c in ["country", "agent", "company", "is_canceled", "reservation_status", "lead_time"]]).show()
print("children = 'NA' rows:", data.filter(F.col("children") == "NA").count())
dup_rows = raw_rows - data.dropDuplicates().count()
zero_night_rows = data.filter((F.col("stays_in_weekend_nights") + F.col("stays_in_week_nights")) == 0).count()
no_guest_rows = data.filter((F.col("adults") + F.coalesce(F.col("children").cast("int"), F.lit(0)) + F.col("babies")) == 0).count()
print("duplicate rows:", dup_rows)
print("zero-night stays (invalid travel time):", zero_night_rows)
print("bookings with no guests:", no_guest_rows)
data.groupBy("reservation_status", "is_canceled").count().show()

df = data.dropDuplicates()
df = df.filter(F.col("is_canceled").isin(0, 1))
df = df.filter(F.col("reservation_status").isNotNull())
df = df.withColumn("children", F.coalesce(F.col("children").cast("int"), F.lit(0)))
df = df.withColumn("country", F.coalesce(F.col("country"), F.lit("UNK")))
df = df.withColumn("has_agent", F.col("agent").isNotNull().cast("int"))
df = df.withColumn("has_company", F.col("company").isNotNull().cast("int"))
df = df.withColumn("total_nights", F.col("stays_in_weekend_nights") + F.col("stays_in_week_nights"))
df = df.filter(F.col("total_nights") > 0)
df = df.filter((F.col("adults") + F.col("children") + F.col("babies")) > 0)
df = df.filter((F.col("adr") >= 0) & (F.col("adr") <= 1000))
df = df.drop("agent", "company", "reservation_status", "reservation_status_date", "assigned_room_type", "required_car_parking_spaces")
df = df.cache()
clean_rows = df.count()
noshow_rate = df.agg(F.avg("is_canceled")).first()[0]
print("rows after cleaning:", clean_rows)
print("overall no-show/cancel rate:", round(noshow_rate, 4))

df = df.withColumnRenamed("deposit_type", "ticket_type")
df = df.withColumn("booking_window", F.when(F.col("lead_time") <= 7, "0-7 days").when(F.col("lead_time") <= 30, "8-30 days").when(F.col("lead_time") <= 90, "31-90 days").when(F.col("lead_time") <= 180, "91-180 days").otherwise("181+ days"))
df.groupBy("booking_window").agg(F.count("*").alias("bookings"), F.round(F.avg("is_canceled"), 3).alias("noshow_rate")).orderBy("noshow_rate").show()

dims = ["hotel", "customer_type", "ticket_type", "booking_window", "market_segment", "distribution_channel"]
eda = reduce(lambda a, b: a.unionByName(b), [df.groupBy(d).agg(F.count("*").alias("bookings"), F.round(F.avg("is_canceled") * 100, 2).alias("noshow_pct")).withColumnRenamed(d, "segment").withColumn("dimension", F.lit(d)) for d in dims])
eda.select("dimension", "segment", "bookings", "noshow_pct").toPandas().to_csv(out_dir + "/eda_segments.csv", index=False)
monthly = df.withColumn("month_num", F.expr("month(to_date(arrival_date_month, 'MMMM'))")).groupBy("arrival_date_year", "month_num").agg(F.count("*").alias("bookings"), F.round(F.avg("is_canceled") * 100, 2).alias("noshow_pct")).orderBy("arrival_date_year", "month_num")
monthly.toPandas().to_csv(out_dir + "/eda_monthly.csv", index=False)

indexer = StringIndexer(inputCol="customer_type", outputCol="customer_type_indexed")
indexer_model = indexer.fit(df)
indexer_data = indexer_model.transform(df)
indexer_data.select("customer_type", "customer_type_indexed").distinct().show()
ohe_demo = OneHotEncoder(inputCols=["customer_type_indexed"], outputCols=["customer_type_vec"])
ohe_demo.fit(indexer_data).transform(indexer_data).select("customer_type", "customer_type_indexed", "customer_type_vec").dropDuplicates(["customer_type"]).show(truncate=False)

train, test = df.randomSplit([0.8, 0.2], seed=42)
train_rows = train.count()
test_rows = test.count()
print("train rows:", train_rows, "| test rows:", test_rows)
print("split check (train + test = clean rows):", train_rows + test_rows == clean_rows)
top = [r["country"] for r in train.groupBy("country").count().orderBy(F.desc("count")).limit(15).collect()]
train = train.withColumn("country_grp", F.when(F.col("country").isin(top), F.col("country")).otherwise("OTHER"))
test = test.withColumn("country_grp", F.when(F.col("country").isin(top), F.col("country")).otherwise("OTHER"))
train = train.withColumn("label", F.col("is_canceled").cast("double"))
test = test.withColumn("label", F.col("is_canceled").cast("double"))

cat_cols = ["ticket_type", "booking_window", "customer_type", "hotel", "market_segment", "distribution_channel", "meal", "reserved_room_type", "arrival_date_month", "country_grp"]
num_cols = ["lead_time", "total_nights", "adults", "children", "babies", "is_repeated_guest", "previous_cancellations", "previous_bookings_not_canceled", "booking_changes", "days_in_waiting_list", "adr", "total_of_special_requests", "arrival_date_week_number", "has_agent", "has_company"]
indexers = [StringIndexer(inputCol=c, outputCol=c + "_idx", handleInvalid="keep") for c in cat_cols]
encoder = OneHotEncoder(inputCols=[c + "_idx" for c in cat_cols], outputCols=[c + "_ohe" for c in cat_cols])
num_assembler = VectorAssembler(inputCols=num_cols, outputCol="num_vec")
scaler = StandardScaler(inputCol="num_vec", outputCol="num_scaled", withMean=True, withStd=True)
final_assembler = VectorAssembler(inputCols=["num_scaled"] + [c + "_ohe" for c in cat_cols], outputCol="features")
prep = Pipeline(stages=indexers + [encoder, num_assembler, scaler, final_assembler]).fit(train)
train_p = prep.transform(train).cache()
test_p = prep.transform(test).cache()
train_p.select("lead_time", "adr", "num_scaled").show(3, truncate=45)
train_p.select("features").show(3, truncate=60)
print("feature vector length:", len(train_p.first()["features"]))

lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=100, regParam=0.001, standardization=False)
lr_model = lr.fit(train_p)
print("intercept:", round(lr_model.intercept, 4), "| number of coefficients:", len(lr_model.coefficients))
predictions = lr_model.transform(test_p)
print("test rows used for evaluation:", predictions.count(), "| expected:", test_rows)
predictions.select("label", "prediction", "probability").show(5, truncate=False)

acc_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = acc_eval.evaluate(predictions)
auc_eval = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
roc_auc = auc_eval.evaluate(predictions)
mc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricLabel=1.0)
precision = mc.setMetricName("precisionByLabel").evaluate(predictions)
recall = mc.setMetricName("recallByLabel").evaluate(predictions)
f1 = mc.setMetricName("fMeasureByLabel").evaluate(predictions)
baseline_acc = test.filter("label = 0").count() / test.count()
print("Accuracy = %g" % accuracy)
print("Test Error = %g" % (1.0 - accuracy))
print("ROC-AUC = %g" % roc_auc)
print("Precision (no-show class) = %g" % precision)
print("Recall    (no-show class) = %g" % recall)
print("F1        (no-show class) = %g" % f1)
print("Baseline accuracy (always predict 'shows up') = %g" % baseline_acc)
predictions.groupBy("label", "prediction").count().orderBy("label", "prediction").show()

predictions = predictions.withColumn("noshow_prob", vector_to_array("probability")[1])
total = train_p.count()
pos = train_p.filter("label = 1").count()
train_w = train_p.withColumn("weight", F.when(F.col("label") == 1, total / (2.0 * pos)).otherwise(total / (2.0 * (total - pos))))
lr_w_model = LogisticRegression(featuresCol="features", labelCol="label", weightCol="weight", maxIter=100, regParam=0.001, standardization=False).fit(train_w)
both = lr_w_model.transform(predictions.withColumnRenamed("rawPrediction", "raw_std").withColumnRenamed("probability", "prob_std").withColumnRenamed("prediction", "pred_std"))
both = both.withColumn("noshow_prob_weighted", vector_to_array("probability")[1])
w_acc = acc_eval.evaluate(both)
w_auc = auc_eval.evaluate(both)
w_precision = mc.setMetricName("precisionByLabel").evaluate(both)
w_recall = mc.setMetricName("recallByLabel").evaluate(both)
w_f1 = mc.setMetricName("fMeasureByLabel").evaluate(both)
print("class-weighted model -> accuracy: %g | recall: %g | precision: %g | ROC-AUC: %g" % (w_acc, w_recall, w_precision, w_auc))

labels = [m.labelsArray[0] for m in prep.stages[:len(cat_cols)]]
feat_names = num_cols + [c + "=" + l for c, ls in zip(cat_cols, labels) for l in ls]
coef = pd.DataFrame({"feature": feat_names, "coefficient": lr_model.coefficients.toArray()})
coef["odds_ratio"] = np.exp(coef["coefficient"])
coef = coef.reindex(coef["coefficient"].abs().sort_values(ascending=False).index)
print(coef.head(10).round(3).to_string(index=False))

pred_cols = ["hotel", "customer_type", "ticket_type", "booking_window", "market_segment", "distribution_channel", "arrival_date_month", "lead_time", "total_nights", "adr", "label", "pred_std", "noshow_prob", "prediction", "noshow_prob_weighted"]
both.select(*pred_cols).withColumnRenamed("label", "actual_noshow").withColumnRenamed("pred_std", "predicted_noshow").withColumnRenamed("prediction", "predicted_noshow_weighted").toPandas().to_csv(out_dir + "/predictions.csv", index=False)
pd.DataFrame({"metric": ["raw_rows", "duplicate_rows", "zero_night_rows", "no_guest_rows", "clean_rows", "noshow_rate_pct", "train_rows", "test_rows", "accuracy", "roc_auc", "precision", "recall", "f1", "baseline_accuracy"], "value": [raw_rows, dup_rows, zero_night_rows, no_guest_rows, clean_rows, noshow_rate * 100, train_rows, test_rows, accuracy, roc_auc, precision, recall, f1, baseline_acc]}).to_csv(out_dir + "/metrics.csv", index=False)
pd.DataFrame({"model": ["Logistic Regression (standard)", "Logistic Regression (class-weighted)"], "accuracy": [accuracy, w_acc], "roc_auc": [roc_auc, w_auc], "precision": [precision, w_precision], "recall": [recall, w_recall], "f1": [f1, w_f1]}).to_csv(out_dir + "/model_comparison.csv", index=False)
predictions.groupBy("label", "prediction").count().withColumn("actual", F.when(F.col("label") == 1, "No-show").otherwise("Shows up")).withColumn("predicted", F.when(F.col("prediction") == 1, "No-show").otherwise("Shows up")).select("actual", "predicted", "count").toPandas().to_csv(out_dir + "/confusion_matrix.csv", index=False)
coef.to_csv(out_dir + "/feature_coefficients.csv", index=False)
print(sorted(os.listdir(out_dir)))
