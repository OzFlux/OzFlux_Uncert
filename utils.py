#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 11:20:56 2018

@author: ian
"""

def rename_df(df, external_names, internal_names):
    assert sorted(external_names.keys()) == sorted(internal_names.keys())
    swap_dict = {external_names[key]: internal_names[key] 
                 for key in internal_names.keys()}
    sub_df = df[swap_dict.keys()].copy()
    sub_df.columns = swap_dict.values()
    return sub_df