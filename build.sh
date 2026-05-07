#!/usr/bin/env bash
set -o errexit
pip install -r requirements.txt
python manage.py collectstatic --noinput
python manage.py migrate
case "${LOAD_INITIAL_DATA:-False}" in
  True|true|1|yes|YES)
    python manage.py load_postgres_data
    ;;
esac
