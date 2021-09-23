import logging
import time

import requests
import torch

from src.constants import classes, landing_statuses, switcher
from src.detected_object import DetectedObject


class ObjectDetectionModel:
    # Base class for team models

    def __init__(self, evaluation_server_url):
        logging.info('Created Object Detection Model')
        self.evaulation_server = evaluation_server_url
        # Modelinizi bu kısımda init edebilirsiniz.

        # self.model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True)
        self.model = torch.hub.load('src/models/yolov5', 'custom',
                                    path='src/models/best.pt',
                                    source='local')  # default

    @staticmethod
    def download_image(img_url, images_folder):
        t1 = time.perf_counter()
        img_bytes = requests.get(img_url).content
        image_name = img_url.split("/")[-1]  # frame_x.jpg

        with open(images_folder + image_name, 'wb') as img_file:
            img_file.write(img_bytes)

        t2 = time.perf_counter()

        logging.info(f'{img_url} - Download Finished in {t2 - t1} seconds to {images_folder + image_name}')

    @staticmethod
    def convert_to_teknofest_model(cls_no, conf):
        if cls_no < 12:
            return switcher.get(cls_no)
        elif cls_no == 14:
            return switcher.get(15)
        else:
            if conf < 0.35:
                if cls_no == 12:
                    return switcher.get(12)
                elif cls_no == 13:
                    return switcher.get(14)
            else:
                if cls_no == 12:
                    return switcher.get(13)
                elif cls_no == 13:
                    return switcher.get(16)

    def object_on_field(self,field_values , all_values):
        for i in all_values:
            width = float(i[2]) - float(i[0])
            height = float(i[1]) - float(i[3])
            if i[5] != 12 or i[5] != 13:
                if float(i[0]) >= float(field_values[0]) and float(i[1]) >= float(field_values[1]):
                    return self.return_classes(field_values[5],False)
                elif float(i[2]) <= float(field_values[2]) and float(i[3]) <= float(field_values[3]):
                    return self.return_classes(field_values[5],False)
                elif float(i[0]) + float(width) and float(i[1]) >= float(field_values[1]):
                    return self.return_classes(field_values[5],False)
                elif float(i[2]) <= float(field_values[2]) and float(height) + float(field_values[3]):
                    return self.return_classes(field_values[5],False)

        field_width = field_values[2] - field_values[0]
        field_height = field_values[1] - field_values[3]

        if field_height <= field_width:
            if field_height / field_width >= 0.7:
                return self.return_classes(field_values[5], True)
            else:
                return self.return_classes(field_values[5],False)
        elif field_width < field_height:
            if field_width / field_height >= 0.7:
                return self.return_classes(field_values[5], True)
            else:
                return self.return_classes(field_values[5], False)


    @staticmethod
    def return_classes(cls_no,con):
        if cls_no == 12:
            if con:
                return switcher.get(13)
            else:
                return switcher.get(12)
        elif cls_no == 13:
            if con:
                return switcher.get(16)
            else:
                return switcher.get(14)


    def process(self, prediction):
        # Yarışmacılar resim indirme, pre ve post process vb işlemlerini burada gerçekleştirebilir.
        # Download image (Example)
        self.download_image(prediction.image_url, "./_images/")

        print(prediction.detected_objects)
        # Örnek: Burada OpenCV gibi bir tool ile preprocessing işlemi yapılabilir. (Tercihe Bağlı)
        # ...
        # Nesne tespiti modelinin bulunduğu fonksiyonun (self.detect() ) çağırılması burada olmalıdır.
        frame_results = self.detect(prediction)
        # Tahminler objesi FramePrediction sınıfında return olarak dönülmelidir.
        return frame_results

    def test_detect(self, image_name):
        results = self.model("./_images/" + image_name, size=1280)

        results.save()

        print(results.pandas().xyxy[0])

        for i in results.pandas().xyxy[0].to_numpy():
            if i[5] == 12 or i[5] == 13:
                if i[4] >= 0.4:
                    dic = self.object_on_field(i, results.pandas().xyxy[0].to_numpy())
                else:
                    continue
            else:
                if i[5] == 2:
                    continue
                else:
                    if i[5] == 14 and i[4] >= 0.4:
                        dic = self.convert_to_teknofest_model(i[5], i[4])
                    elif i[5] == 5 and i[4] >= 0.4:
                        dic = self.convert_to_teknofest_model(i[5], i[4])
                    elif i[5] != 14 and i[5] != 5:
                        dic = self.convert_to_teknofest_model(i[5], i[4])
                    else:
                        continue
            print(dic)
            print(i)

    def detect(self, prediction):
        image_name = prediction.image_url.split("/")[-1]  # frame_x.jpg

        results = self.model("./_images/" + image_name, size=1280)

        # results.print()
        results.save()

        print(results.pandas().xyxy[0])

        for i in results.pandas().xyxy[0].to_numpy():
            if i[5] == 12 or i[5] == 13:
                if i[4] >= 0.4:
                    dic = self.object_on_field(i, results.pandas().xyxy[0].to_numpy())
                else:
                    continue
            else:
                if i[5] == 2:
                    continue
                else:
                    if i[5] == 14 and i[4] >= 0.3:
                        dic = self.convert_to_teknofest_model(i[5], i[4])
                    elif i[5] == 5 and i[4] >= 0.3:
                        dic = self.convert_to_teknofest_model(i[5], i[4])
                    elif i[5] != 14 and i[5] != 5:
                        dic = self.convert_to_teknofest_model(i[5], i[4])
                    else:
                        continue
            print(dic)
            print(i)
            # cls = classes["UAP"],  # Tahmin edilen nesnenin sınıfı classes sözlüğü kullanılarak atanmalıdır.
            cls = dic["classes"]
            landing_status = dic["landing_statuses"]
            # landing_status = landing_statuses["Inilebilir"]  # Tahmin edilen nesnenin inilebilir durumu landing_statuses sözlüğü kullanılarak atanmalıdır
            top_left_x = i[0]
            top_left_y = i[1]
            bottom_right_x = i[2]
            bottom_right_y = i[3]
            d_obj = DetectedObject(cls,
                                   landing_status,
                                   top_left_x,
                                   top_left_y,
                                   bottom_right_x,
                                   bottom_right_y)
            print(d_obj)
            # Modelin tahmin ettiği her nesne prediction nesnesi içerisinde bulunan detected_objects listesine eklenmelidir.
            prediction.add_detected_object(d_obj)
        return prediction
