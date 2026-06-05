/*
######################################################################
# TIE-02402 Ohjelmointi 3: Perusteet, S2019                          #
#                                                                    #
# Project: Colonizing Pirkanmaa                                      #
# Program description: Program instructions are located in           #
#                      Documentation/documentation.pdf               #
#                                                                    #
# File: basicresources.cpp, see basicresources.h for more info       #
#                                                                    #
# Authors: Otto Ranta-Ojala, 253561, otto.ranta-ojala@tuni.fi        #
#          Tino Kaartovuori, 254987, tino.kaartovuori@tuni.fi        #
######################################################################
*/


#include "basicresources.h"
#include <QDebug>

namespace Course {


ResourceMap mergeResourceMaps(const ResourceMap& left_,
                              const ResourceMap& right_)
{

    ResourceMap new_map = right_;

    for( auto it = left_.begin(); it != left_.end(); ++it)
    {
        if(new_map.find(it->first) == new_map.end())
        {
            new_map[it->first] = it->second;
        }
        else
        {
            new_map[it->first] = new_map[it->first] + it->second;
        }
    }

    return new_map;
}

ResourceMap reverseResourceMap(const ResourceMap& map)
{
    ResourceMap map_ = map;

    for ( auto it = map_.begin(); it != map_.end(); ++it)
    {
        map_[it->first] = it->second * (-1);
    }

    return map_;
}


ResourceMap getNegativesMap(const ResourceMap& map) {
    ResourceMap map_ = {};

    for ( auto it = map.begin(); it != map.end(); ++it)
    {
        if (it->second < 0) {
            map_[it->first] = it->second;
        } else {
            map_[it->first] = 0;
        }
    }

    return map_;
}


ResourceMap getPositivesMap(const ResourceMap &map)
{
    ResourceMap map_ = {};

    for ( auto it = map.begin(); it != map.end(); ++it)
    {

        if (it->second >= 0) {
            map_[it->first] = it->second;
        } else {
            map_[it->first] = 0;
        }
    }

    return map_;
}

}
