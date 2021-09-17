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

        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

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
    def convert_to_teknofest_model(cls_no,conf):
        if cls_no<12:
            return switcher.get(cls_no)
        else:
            if conf<0.5:
                if cls_no == 11:
                    return switcher.get(12)
                else:
                    return switcher.get(14)
            else:
                if cls_no == 11:
                    return switcher.get(13)
                else:
                    return switcher.get(15)



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

    def detect(self, prediction):
        image_name = prediction.image_url.split("/")[-1]  # frame_x.jpg

        results = self.model("./_images/" + image_name)

        results.print()
        results.save()

        print(results.pandas().xyxy[0])

        for i in results.pandas().xyxy[0].to_numpy():
            dic = self.convert_to_teknofest_model(i[6],i[7])
            print(i)
            # cls = classes["UAP"],  # Tahmin edilen nesnenin sınıfı classes sözlüğü kullanılarak atanmalıdır.
            cls = dic["classes"]
            landing_status = dic["landing_statuses"]
            # landing_status = landing_statuses["Inilebilir"]  # Tahmin edilen nesnenin inilebilir durumu landing_statuses sözlüğü kullanılarak atanmalıdır
            top_left_x = i[1]
            top_left_y = i[2]
            bottom_right_x = i[3]
            bottom_right_y = i[4]
            d_obj = DetectedObject(cls,
                                   landing_status,
                                   top_left_x,
                                   top_left_y,
                                   bottom_right_x,
                                   bottom_right_y)
            print(d_obj)
            # Modelin tahmin ettiği her nesne prediction nesnesi içerisinde bulunan detected_objects listesine eklenmelidir.
            prediction.add_detected_object(d_obj)


        # Modelinizle bu fonksiyon içerisinde tahmin yapınız.
        # results = self.model.evaluate(...) # Örnektir.
        # Burada örnek olması amacıyla 20 adet tahmin yapıldığı simüle edilmiştir.
        # Yarışma esnasında modelin tahmin olarak ürettiği sonuçlar kullanılmalıdır.
        # Örneğin :
        # for i in results: # gibi
        # for i in range(1, 20):
        #     cls = classes["UAP"],  # Tahmin edilen nesnenin sınıfı classes sözlüğü kullanılarak atanmalıdır.
        #     landing_status = landing_statuses["Inilebilir"]  # Tahmin edilen nesnenin inilebilir durumu landing_statuses sözlüğü kullanılarak atanmalıdır.
        #     top_left_x = 12 * i  # Örnek olması için rastgele değer atanmıştır. Modelin sonuçları kullanılmalıdır.
        #     top_left_y = 12 * i  # Örnek olması için rastgele değer atanmıştır. Modelin sonuçları kullanılmalıdır.
        #     bottom_right_x = 12 * i  # Örnek olması için rastgele değer atanmıştır. Modelin sonuçları kullanılmalıdır.
        #     bottom_right_y = 12 * i  # Örnek olması için rastgele değer atanmıştır. Modelin sonuçları kullanılmalıdır.
        #
        #     # Modelin tespit ettiği her bir nesne için bir DetectedObject sınıfına ait nesne oluşturularak
        #     # tahmin modelinin sonuçları parametre olarak verilmelidir.
        #     d_obj = DetectedObject(cls,
        #                            landing_status,
        #                            top_left_x,
        #                            top_left_y,
        #                            bottom_right_x,
        #                            # bottom_right_y)
        #     # Modelin tahmin ettiği her nesne prediction nesnesi içerisinde bulunan detected_objects listesine eklenmelidir.
        #     prediction.add_detected_object(d_obj)

        return prediction
