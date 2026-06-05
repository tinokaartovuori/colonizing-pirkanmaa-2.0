/*
############################################################################
# TIE-02402 Ohjelmointi 3: Perusteet, S2019                                #
#                                                                          #
# Project: Colonizing Pirkanmaa                                            #
# Program description: Program instructions are located in                 #
#                      Documentation/documentation.pdf                     #
#                                                                          #
# File: headquarters.cpp, see headquarters.h for the class's description   #
#                                                                          #
# Authors: Otto Ranta-Ojala, 253561, otto.ranta-ojala@tuni.fi              #
#          Tino Kaartovuori, 254987, tino.kaartovuori@tuni.fi              #
############################################################################
*/

#include "headquarters.h"
#include "Tiles/tilebase.h"
#include <QDebug>


namespace Course {

HeadQuarters::HeadQuarters(
        const std::weak_ptr<iGameEventHandler>& eventhandler,
        const std::weak_ptr<iObjectManager>& objectmanager,
        const std::weak_ptr<PlayerBase>& owner):
    BuildingBase(eventhandler, objectmanager, owner),
    conqured_(false)
{
    addBasicDescription(Student::ConstDescriptionMaps::HEADQUARTERS_DESCRIPTION);
}

std::string HeadQuarters::getType() const
{
    return "Headquarters";
}

std::string HeadQuarters::getExtraDescription() {
    if (!conqured_) {
        return "<u>Effects:</u><br>+" +
                std::to_string(ConstResourceMaps::HQ_UNIT_VALUE) +
                " Max Units<br>+" +
                std::to_string(ConstResourceMaps::HQ_SOLDIER_VALUE) +
                " Max Soldiers";
    } else {
        return "";
    }

}

void HeadQuarters::setConquered()
{
    conqured_ = true;
    lockEventHandler()->updateAnimatedTileToStatic
            (parentTile_.lock(), 1); //Frame is set to 1 graphically
    setImageFiles(ImageVectors::HEADQUARTERSDESTROYED);
    addBasicDescription
            (Student::ConstDescriptionMaps::BROKEN_HEADQUARTERS_DESCRIPTION);
}

bool HeadQuarters::isConquered()
{
    return conqured_;
}

} // namespace Course
