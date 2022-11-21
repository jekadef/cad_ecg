#!/bin/bash
mysql -h 10.102.25.4 \
  --user jessica.defreitas@icahn.mssm.edu@hpims-mysql \
  --enable-cleartext-plugin \
  --password=`az account get-access-token --resource-type oss-rdbms --output tsv --query accessToken`\
  < ./msdw_query_cIHD_20220209.sql