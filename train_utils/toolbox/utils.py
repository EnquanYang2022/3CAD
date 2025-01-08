
def get_dataset(data_name,data_path = './MvtecAD/',dtd_path='dtd/images/',class_name=None):
    if data_name == 'mvtec2d':
        from .datasets.MvtecAD import MVTec2D
        return MVTec2D(data_path=data_path,dtd_path=dtd_path,class_name=class_name,phase='train'), \
               MVTec2D(data_path=data_path,dtd_path=dtd_path,class_name=class_name,phase='test')
    elif data_name == '3CAD':
        from .datasets.Data_3CAD import Data_3CAD
        return Data_3CAD(data_path=data_path,dtd_path=dtd_path,class_name=class_name,phase='train'), \
               Data_3CAD(data_path=data_path,dtd_path=dtd_path,class_name=class_name,phase='test')


def get_model(model_name):
    if model_name == 'cfrg':
        from .models.cfrg.model import cfrg_net
        return cfrg_net()





