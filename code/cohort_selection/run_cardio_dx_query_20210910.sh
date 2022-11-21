#!/bin/bash
mysql -h 10.102.25.4 \
  --user jessica.defreitas@mtsinai.onmicrosoft.com@hpims-mysql \
  --enable-cleartext-plugin \
  --password=`az account get-access-token --resource-type oss-rdbms --output tsv --query accessToken`\
  < ./cardio_dx_query_20210910.sql