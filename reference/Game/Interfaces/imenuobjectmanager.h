/*
######################################################################
# TIE-02402 Ohjelmointi 3: Perusteet, S2019                          #
#                                                                    #
# Project: Colonizing Pirkanmaa                                      #
# Program description: Program instructions are located in           #
#                      Documentation/documentation.pdf               #
#                                                                    #
# File: imenuobjectMḿanager.h, interface for MenuObjectManager       #
#                                                                    #
# Authors: Otto Ranta-Ojala, 253561, otto.ranta-ojala@tuni.fi        #
#          Tino Kaartovuori, 254987, tino.kaartovuori@tuni.fi        #
######################################################################
*/

#ifndef IMENUOBJECTMANAGER_H
#define IMENUOBJECTMANAGER_H

#include <memory>
#include <vector>

#include "Menus/menuview.h"
#include "Core/coordinate.h"
#include "Menus/button.h"
#include "Menus/label.h"


namespace Student {

class ObjectManager;
class GameSettingsManager;
class PlayerManager;
class GameScene;

#ifndef COURSE_OBJECTID
#define COURSE_OBJECTID
using ObjectId = unsigned int;
#endif

/**
 * @brief The iMenuObjectManager class is an interface which the Course-side
 * code uses to interact with the ObjectManager implemented by the students.
 *
 * @note The interface declares only functions required by the Course-side code.
 * The actual implementation can (and should!) contain more stuff.
 */
class iMenuObjectManager
{
public:

    /**
     * @brief Default destructor.
     *
     */

    virtual ~iMenuObjectManager() = default;

    virtual void setGameScene(std::shared_ptr<GameScene> gs) = 0;

    virtual void addDALS(const std::shared_ptr<Course::iGameEventHandler> gameeventhandler,
                 const std::shared_ptr<Student::ObjectManager> objectmanager,
                 const std::shared_ptr<Student::PlayerManager> playermanager,
                 const std::shared_ptr<Student::GameSettingsManager> gamesettingsmanager) = 0;

    virtual void setTileInspectionMenuView(std::shared_ptr<Course::TileBase> tile, int index_for_buildings = 0) = 0;

    virtual void setDefaultMenuView() = 0;

    virtual void setStatMenuView() = 0;

    virtual void selectFirstTileMenuView(std::shared_ptr<Course::PlayerBase> player) = 0;

    virtual void resetMenuView() = 0;

    virtual void setPlayerLostMenu(std::vector<std::shared_ptr<Course::PlayerBase> > players, std::vector<std::string> reasons) = 0;

    virtual void setTieMenu(std::vector<std::shared_ptr<Course::PlayerBase> > players, std::vector<std::string> reasons) = 0;

    virtual void setWinMenu(std::shared_ptr<Course::PlayerBase> player) = 0;

    virtual void addPlayerTitle() = 0;

    virtual void setUnitShopMenuView() = 0;

    virtual void addCloseButton() = 0;

    virtual void addResourceMenu(int x, int y) = 0;

    virtual void addNetMenu(int x, int y) = 0;

    virtual void addUnitMenu(int x, int y) = 0;

    virtual void addOKButton(int x, int y) = 0;

    virtual void addTilePercents(int x, int y) = 0;

    virtual void addRoundsPlayed(int x, int y) = 0;

}; // class iMenuObjectManager

} // namespace Course


#endif // IMENUOBJECTMANAGER_H
