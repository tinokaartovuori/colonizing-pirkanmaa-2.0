/*
######################################################################
# TIE-02402 Ohjelmointi 3: Perusteet, S2019                          #
#                                                                    #
# Project: Colonizing Pirkanmaa                                      #
# Program description: Program instructions are located in           #
#                      Documentation/documentation.pdf               #
#                                                                    #
# File: basicresource.h, header for BasicResource-class              #
#                                                                    #
# Authors: Otto Ranta-Ojala, 253561, otto.ranta-ojala@tuni.fi        #
#          Tino Kaartovuori, 254987, tino.kaartovuori@tuni.fi        #
######################################################################
*/


#ifndef BASICRESOURCES_H
#define BASICRESOURCES_H

#include <map>
namespace Course {

/**
 * @brief BasicResource is an enumeration for different basic resource-types
 */
enum BasicResource {
    NONE = 0,
    MONEY = 1,
    WOOD = 2,
    STONE = 3,
    METAL = 4,
};

/**
 * @brief ResourceMap is an alias for std::map<BasicResource, int>
 */
using ResourceMap = std::map<BasicResource, int>;


/**
 * @brief Creates a new ResourceMap that contains summed values of two
 * ResourceMaps
 * @param left first ResourceMap
 * @param right second ResourceMap
 * @return ResourceMap that has summed values from the two ResourceMaps
 */
ResourceMap mergeResourceMaps(const ResourceMap& left_,
                              const ResourceMap& right_);

ResourceMap reverseResourceMap(const ResourceMap& map);

ResourceMap getNegativesMap(const ResourceMap& map);

ResourceMap getPositivesMap(const ResourceMap& map);

}

#endif // BASICRESOURCES_H
