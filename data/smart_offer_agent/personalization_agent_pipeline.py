# Databricks notebook source
dbutils.widgets.text("train_run", "False")

dbutils.notebook.run("/Repos/shared_repo/xxxrx-advanced-analytics-databricks/3 - gold/personalized_offers/personalization_smart_offering_metrics", 15800)
dbutils.notebook.run("/Repos/shared_repo/xxxrx-advanced-analytics-databricks/3 - gold/personalized_offers/personalization_cagm_state", 15800, {'train_run': dbutils.widgets.get('train_run')})
dbutils.notebook.run("/Repos/shared_repo/xxxrx-advanced-analytics-databricks/3 - gold/personalized_offers/personalization_reward_table", 15800)
dbutils.notebook.run("/Repos/shared_repo/xxxrx-advanced-analytics-databricks/3 - gold/personalized_offers/personalization_agent_process_raw_data", 15800)
dbutils.notebook.run("/Repos/shared_repo/xxxrx-advanced-analytics-databricks/3 - gold/personalized_offers/personalization_agent_data", 15800)
#dbutils.notebook.run("/Repos/shared_repo/xxxrx-advanced-analytics-databricks/3 - gold/personalized_offers/personalization_performance_report", 15800)
