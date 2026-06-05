/*
############################################################################
# TIE-02402 Ohjelmointi 3: Perusteet, S2019                                #
#                                                                          #
# Project: Colonizing Pirkanmaa                                            #
# Program description: Program instructions are located in                 #
#                      Documentation/documentation.pdf                     #
#                                                                          #
# File: basicworker.cpp, see basicworker.h for the class's description     #
#                                                                          #
# Authors: Otto Ranta-Ojala, 253561, otto.ranta-ojala@tuni.fi              #
#          Tino Kaartovuori, 254987, tino.kaartovuori@tuni.fi              #
############################################################################
*/


#include "basicworker.h"
#include "Tiles/tilebase.h"
#include "Interfaces/igameeventhandler.h"
#include "Interfaces/iobjectmanager.h"

namespace Course {



BasicWorker::BasicWorker(const std::weak_ptr<iGameEventHandler>& eventhandler,
                         const std::weak_ptr<iObjectManager>& objectmanager,
                         const std::weak_ptr<Student::GameSettingsManager>& gamesettingsmanager,
                         const std::weak_ptr<PlayerBase>& owner,
                         const std::weak_ptr<Course::TileBase>& tile
 ):
    UnitBase(
        eventhandler,
        objectmanager,
        gamesettingsmanager,
        owner,
        tile)
{
}

BasicWorker::BasicWorker(const std::weak_ptr<iGameEventHandler> &eventhandler,
                         const std::weak_ptr<iObjectManager> &objectmanager,
                         const std::weak_ptr<Student::GameSettingsManager> &gamesettingsmanager,
                         const std::weak_ptr<PlayerBase> &owner):
    UnitBase(
        eventhandler,
        objectmanager,
        gamesettingsmanager,
        owner)
{
}

std::string BasicWorker::getType() const
{
    return "BasicWorker";
}


ResourceMap BasicWorker::getSalary()
{
    return ConstResourceMaps::BASIC_WORKER_SALARY;
}

ResourceMap BasicWorker::getCost()
{
    return ConstResourceMaps::BASIC_WORKER_COST;
}



} // namespace Course
