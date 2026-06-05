/*
######################################################################
# TIE-02402 Ohjelmointi 3: Perusteet, S2019                          #
#                                                                    #
# Project: Colonizing Pirkanmaa                                      #
# Program description: Program instructions are located in           #
#                      Documentation/documentation.pdf               #
#                                                                    #
# File: menuobjectmanager.h, header to the MenuObjectManager-class   #
#                                                                    #
# Authors: Otto Ranta-Ojala, 253561, otto.ranta-ojala@tuni.fi        #
#          Tino Kaartovuori, 254987, tino.kaartovuori@tuni.fi        #
######################################################################
*/

#ifndef MENUOBJECTMANAGER_H
#define MENUOBJECTMANAGER_H

#include <memory>
#include <vector>

#include "Interfaces/imenuobjectmanager.h"

#include "DAL/objectmanager.h"
#include "Interfaces/igameeventhandler.h"
#include "DAL/gamesettingsmanager.h"
#include "DAL/playermanager.h"

#include "Graphics/gamescene.h"
#include "Core/resourcemaps.h"
#include "Core/descriptionmaps.h"

#include "Tiles/forest.h"
#include "Tiles/grassland.h"
#include "Tiles/mountain.h"
#include "Tiles/river.h"
#include "Tiles/tilebase.h"

namespace Student {

/**
 * @brief The MenuObjectManager class is an interface which the Course-side
 * code uses to interact with the MenuObjectManager implemented by the students.
 *
 * @note The interface declares only functions required by the Course-side code.
 * The actual implementation can (and should!) contain more stuff.
 */
class MenuObjectManager : public Student::iMenuObjectManager
{
public:

    MenuObjectManager();
    /**
     * @brief Default destructor.
     */
    ~MenuObjectManager() override;

    void setGameScene(std::shared_ptr<GameScene> gs) override;

    void addDALS(const std::shared_ptr<Course::iGameEventHandler> gameeventhandler,
           const std::shared_ptr<Student::ObjectManager> objectmanager,
           const std::shared_ptr<PlayerManager> playermanager,
           const std::shared_ptr<Student::GameSettingsManager> gamesettingsmanager) override;

    void setTileInspectionMenuView(std::shared_ptr<Course::TileBase> tile, int index_for_buildings = 0) override;

    void setDefaultMenuView() override;

    void setStatMenuView() override;

    void selectFirstTileMenuView(std::shared_ptr<Course::PlayerBase> player) override;

    void resetMenuView() override;

    void setPlayerLostMenu(std::vector<std::shared_ptr<Course::PlayerBase> > players, std::vector<std::string> reasons) override;

    void setTieMenu(std::vector<std::shared_ptr<Course::PlayerBase> > players, std::vector<std::string> reasons) override;

    void setWinMenu(std::shared_ptr<Course::PlayerBase> player) override;

    void addPlayerTitle() override;

    void setUnitShopMenuView() override;

    void addResourceMenu(int x, int y) override;

    void addNetMenu(int x, int y) override;

    void addUnitMenu(int x, int y) override;

    void addCloseButton() override;

    void addOKButton(int x, int y) override;

    void addTilePercents(int x, int y) override;

    void addRoundsPlayed(int x, int y) override;  

private:

    std::weak_ptr<GameScene> gameScene_;
    std::weak_ptr<Student::GameSettingsManager> gameSettingsManager_;
    std::weak_ptr<Student::PlayerManager> playerManager_;
    std::weak_ptr<Course::iGameEventHandler> gameEventHandler_;
    std::weak_ptr<Student::ObjectManager> objectManager_;
    std::shared_ptr<Student::MenuView> currentMenuView_;


}; // class MenuObjectManager

} // namespace Student


#endif // MENUOBJECTMANAGER_H
