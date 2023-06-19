.PHONY: env, download_data

env:
	mamba env create --prefix ./env -f environment.yaml

download_data:
	wget -nv -O data.zip https://portal.ijs.si/nextcloud/s/bYGaCcemSJa3mYk/download
	unzip -q data.zip
	rm data.zip


