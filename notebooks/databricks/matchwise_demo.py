# Databricks Free Edition extension (source-exportable, no credentials required).
# Run after uploading the demo CSV to a workspace volume.
from pyspark.sql import functions as F

# Databricks provides the `spark` session implicitly.
# ruff: noqa: F821
bronze = spark.read.option("header", True).csv("/Volumes/<catalog>/<schema>/catalog.csv")
silver = bronze.withColumn("brand_norm", F.lower(F.trim("brand"))).withColumn("title_norm", F.lower(F.trim("title")))
silver.write.format("delta").mode("overwrite").saveAsTable("matchwise_silver_catalog")
spark.sql("SELECT brand_norm, count(*) AS products FROM matchwise_silver_catalog GROUP BY brand_norm ORDER BY products DESC").show()
