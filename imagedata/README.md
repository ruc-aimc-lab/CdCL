# Data preparation
### Datasets
- RFMiD: [https://ieee-dataport.org/open-access/retinal-fundus-multi-disease-image-dataset-rfmid](https://ieee-dataport.org/open-access/retinal-fundus-multi-disease-image-dataset-rfmid)
- TOP: [https://github.com/DateCazuki/Fundus_Diagnosis](https://github.com/DateCazuki/Fundus_Diagnosis)

### Data Organization
The images are organized as follows:
```
./imagedata/  
    RFMiD/
        train_1.jpg
        train_2.jpg
        ...
        val_1.jpg
        val_2.jpg
        ...
        test_1.jpg
        test_2.jpg
        ...
    TOP/
        000000_00.jpg
        000000_01.jpg
        000001_00.jpg
        ...
```
The annotations are organized as follows:
```
./Annotations/
    test_RFMiD/
        Annotations/
            anno.txt
    test_TOP/
    train_RFMiD/
    train_TOP/
    val_RFMiD/
    val_TOP/
```

