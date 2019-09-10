#coding=utf-8

import os
import json
from geo_API import GeoAPI


def generate_theta(source_image_path, target_image_path, geo):
    # geo = GeoAPI()

    theta_aff, theta_tps, theta_aff_tps = geo.get_thetas(source_image_path, target_image_path)
    theta_aff_list = theta_aff.cpu().data.numpy()[0].tolist()
    theta_tps_list = theta_tps.cpu().data.numpy()[0].tolist()
    theta_aff_tps_list = theta_aff_tps.cpu().data.numpy()[0].tolist()

    sub_json = {}
    sub_json['aff'] = theta_aff_list
    sub_json['tps'] = theta_tps_list
    sub_json['aff_tps'] = theta_aff_tps_list

    return sub_json

