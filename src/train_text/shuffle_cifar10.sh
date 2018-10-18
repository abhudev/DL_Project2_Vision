# Shuffle the dataset once, and create dev and test files.

paste -d '|' cifar10_train_img.txt cifar10_train_classes.txt | shuf | awk -v FS="|" '{ print $1 > "train_img_cifar10.txt" ; print $2 > "train_classes_cifar10.txt" }'

head -10000 train_img_cifar10.txt > dev_img_cifar10.txt
sed -i 1,10000d train_img_cifar10.txt
head -10000 train_classes_cifar10.txt > dev_classes_cifar10.txt
sed -i 1,10000d train_classes_cifar10.txt

paste -d '|' CUB_train_img.txt CUB_train_classes.txt CUB_train_bbox.txt | shuf | awk -v FS="|" '{ print $1 > "train_img_CUB.txt" ; print $2 > "train_classes_CUB.txt" ; print $3 > "train_bbox_CUB.txt" }'

head -600 train_img_CUB.txt > dev_img_CUB.txt
sed -i 1,600d train_img_CUB.txt
head -600 train_classes_CUB.txt > dev_classes_CUB.txt
sed -i 1,600d train_classes_CUB.txt
head -600 train_bbox_CUB.txt > dev_bbox_CUB.txt
sed -i 1,600d train_bbox_CUB.txt
