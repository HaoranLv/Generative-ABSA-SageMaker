all:
	rm -rf public
	hugo --minify
# for f in `find . | grep .html$`; do sed -i -- 's/baseurl\=""/baseurl\="https\:\/\/awslabs\.github\.io\/spot\-tagging\-bot\-for\-digital\-assets\/"/g' $f; done

serve:
	hugo server
