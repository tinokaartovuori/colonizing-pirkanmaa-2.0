/*
######################################################################
# TIE-02402 Ohjelmointi 3: Perusteet, S2019                          #
#                                                                    #
# Project: Colonizing Pirkanmaa                                      #
# Program description: Program instructions are located in           #
#                      Documentation/documentation.pdf               #
#                                                                    #
# File: gameeventhandler.cpp, see gameeventhandler.h for more info   #
#                                                                    #
# Authors: Otto Ranta-Ojala, 253561, otto.ranta-ojala@tuni.fi        #
#          Tino Kaartovuori, 254987, tino.kaartovuori@tuni.fi        #
######################################################################
*/

#include "gameeventhandler.h"
#include "iostream"
#include <QDebug>


namespace Student {

GameEventHandler::GameEventHandler()
{

}

GameEventHandler::GameEventHandler(
          std::shared_ptr<Student::ObjectManager> objectmanager,
          std::shared_ptr<PlayerManager> playermanager,
          std::shared_ptr<Student::MenuObjectManager> menuobjectmanager,
          std::shared_ptr<Student::GameSettingsManager> gamesettingsmanager
          ):
          objectManager_(objectmanager),
          playerManager_(playermanager),
          menuObjectManager_(menuobjectmanager),
          gameSettingsManager_(gamesettingsmanager),
          unitToDeploy_(nullptr)
{
}


void GameEventHandler::setGameScene(std::shared_ptr<GameScene> gs)
{
    gameScene_ = gs;
}


void GameEventHandler::firstRoundActions(std::shared_ptr<Course::TileBase> tile)
{
    // Checks if tile already contains building
    if (tile->getBuilding() != nullptr) return;

    std::shared_ptr<Course::HeadQuarters> HQ =
            std::make_shared<Course::HeadQuarters>
            (shared_from_this(),
             objectManager_,
             playerManager_.lock()->getCurrentPlayer());

    //Selects the correct colour-themed graphics according to the player num
    if (playerManager_.lock()->getCurrentPlayer()->getPlayerNum() == 1) {
        HQ->setImageFiles(ImageVectors::HEADQUARTERSONE);
    }
    else if (playerManager_.lock()->getCurrentPlayer()->getPlayerNum() == 2) {
        HQ->setImageFiles(ImageVectors::HEADQUARTERSTWO);
    }
    else if (playerManager_.lock()->getCurrentPlayer()->getPlayerNum() == 3) {
        HQ->setImageFiles(ImageVectors::HEADQUARTERSTHREE);
    }
    else if (playerManager_.lock()->getCurrentPlayer()->getPlayerNum() == 4) {
        HQ->setImageFiles(ImageVectors::HEADQUARTERSFOUR);
    }

    HQ->setAnimationOption(AnimationOptions::HEADQUARTERS);
    tile->addBuilding(HQ);

    tile->setOwner(playerManager_.lock()->getCurrentPlayer());
    updateTile(tile);

    std::vector<Course::Coordinate> neighboursCoordinates
            = tile->getCoordinatePtr()->neighbours(1,
                            gameSettingsManager_.lock()->getMapGridWidth(),
                            gameSettingsManager_.lock()->getMapGridHeight());

    for(int i=0;i<(int)neighboursCoordinates.size();++i)
    {
        std::shared_ptr<Course::TileBase> neighbour
               = objectManager_.lock()->getTile(neighboursCoordinates.at(i));

        // If the Tile doesn't have owner, set it's owner to buildings owner.
        if( neighbour->getOwner() == nullptr)
        {
            neighbour->setOwner(playerManager_.lock()->getCurrentPlayer());
            if (neighbour->getBuilding() != nullptr) {
                if (neighbour->getBuilding()->getType() == "Mikontalo") {
                    neighbour->getBuilding()
                          ->setOwner(playerManager_.lock()->getCurrentPlayer());
                }
            }
        }
    }


    /*Checks if player has areas that are not connected to the HQ.
     *If there is, they'll become owned by no one.*/
    std::shared_ptr<Course::PlayerBase> player =
            playerManager_.lock()->getCurrentPlayer();
    std::vector<std::shared_ptr<Course::TileBase>> HqConnectedTiles =
            objectManager_.lock()->getHqConnectedTiles(player);

    //Loops all player's tiles.
    for (auto object : player->getObjects()) {
        if (std::dynamic_pointer_cast<Course::TileBase>(object) != nullptr)
        {
            std::shared_ptr<Course::TileBase> tile =
                    std::dynamic_pointer_cast<Course::TileBase>(object);

            //Tile isn't connected to the HQ The tile becomes owned by no one
           if (!(std::find(HqConnectedTiles.begin(),
                                 HqConnectedTiles.end(),
                                 tile) != HqConnectedTiles.end()))
            {
                tile->setOwner(nullptr);
            }
        }
    }


    playerManager_.lock()->changeTurn();

    if (playerManager_.lock()->getCurrentPlayer()->getObjects().size() == 0)
    {
        menuObjectManager_.lock()->selectFirstTileMenuView
                (playerManager_.lock()->getCurrentPlayer());
    } else {
        openDefaultMenuView();
    }

}


void GameEventHandler::tileClicked(std::shared_ptr<Course::TileBase> tile)
{
    //When there is only one player or less. Clicking tiles is not allowed
    if (playerManager_.lock()->getPlayers().size() <= 1) return;

    //Keeps track of the tile that was clicked previously
    Course::Coordinate last_tile_coord = Course::Coordinate(-1, -1);
    if (objectManager_.lock()->getClickedTileBorder() != nullptr) {
        last_tile_coord =
               objectManager_.lock()->getClickedTileBorder()->getCoordinate();
    }
    objectManager_.lock()->removeClickedTileBorder();

    /*When player in turn clicks non-owned grassland firstRoundActions is executed
     *if the player doesn't have own objects*/
    if (playerManager_.lock()->getCurrentPlayer()->getObjects().size() == 0
        && tile->getType() == "Grassland"
        && tile->getOwner() == nullptr)
    {
        firstRoundActions(tile);
        gameScene_.lock()->updateTile(tile);
    }
    else if (playerManager_.lock()->getCurrentPlayer()->getObjects().size() != 0){
        if (unitToDeploy_ != nullptr) {
            //Unit is being moved from a tile
            if (unitPreviousTile_ != nullptr) {
                try {
                    objectManager_.lock()->setClickedTileBorder(unitPreviousTile_);

                    unitToDeploy_->setOwner
                            (playerManager_.lock()->getCurrentPlayer());

                    if (tile->getOwner() !=
                            playerManager_.lock()->getCurrentPlayer())
                    {
                        unitToDeploy_->setAsConquering(true);
                    }
                    else {
                        unitToDeploy_->setAsConquering(false);
                    }

                    tile->addUnit(unitToDeploy_);
                    unitToDeploy_->addParentTile(tile);

                    unitPreviousTile_->removeUnit(unitToDeploy_);

                    tile->updateAnimation();
                    unitPreviousTile_->updateAnimation();

                    gameScene_.lock()->removeMouseFollowItem();
                    objectManager_.lock()->removeBlockTileOverlays();

                    tile->updateUnitCoordinates();
                    unitPreviousTile_->updateUnitCoordinates();

                    gameScene_.lock()->updateTile(tile);
                    gameScene_.lock()->updateTile(unitPreviousTile_);

                    unitToDeploy_ = nullptr;
                    setTileInspectionMenuView(unitPreviousTile_);
                    unitPreviousTile_ = nullptr;
                }
                catch (...) {
                    qDebug() << "Unit cannot be moved there!";
                    return;
                }
            //Unit is not being moved from a different tile
            } else {
                try {
                    if (canBuyUnitOrBuilding(unitToDeploy_)) {
                        unitToDeploy_->addParentTile(tile);
                        unitToDeploy_->setOwner
                                (playerManager_.lock()->getCurrentPlayer());
                        tile->addUnit(unitToDeploy_);
                        buyUnitOrBuilding(unitToDeploy_);
                    }
                    else {
                        unitToDeploy_ = nullptr;
                        unitPreviousTile_ = nullptr;
                        gameScene_.lock()->removeMouseFollowItem();
                        objectManager_.lock()->removeBlockTileOverlays();
                        return;
                    }
                    unitToDeploy_ = nullptr;
                    unitPreviousTile_ = nullptr;
                    gameScene_.lock()->removeMouseFollowItem();
                    objectManager_.lock()->removeBlockTileOverlays();
                    tile->updateAnimation();
                    gameScene_.lock()->updateTile(tile);

                    menuObjectManager_.lock()->setUnitShopMenuView();
                }
                catch (...) {
                    qDebug() << "You cant place bought unit there!";
                    return;
                }

            }

        } else {
            if (last_tile_coord != tile->getCoordinate()) {
                setTileInspectionMenuView(tile);
                objectManager_.lock()->setClickedTileBorder(tile);
            } else {
                openDefaultMenuView();
            }
        }
    }

    //gameScene_->updateTile(tile);

}


void GameEventHandler::endTurn()
{
    std::vector<std::string> losingReasons = {};

    //Loops player's all tiles and generates resources from them.
    for (auto object : playerManager_.lock()->getCurrentPlayer()->getObjects()) {
        if (std::dynamic_pointer_cast<Course::TileBase>(object) != nullptr)
        {
            std::shared_ptr<Course::TileBase> tile =
                    std::dynamic_pointer_cast<Course::TileBase>(object);
            tile->generateResources();
        }
    }

    //Loops player's all units and pays salaries for them.
    for (auto object : playerManager_.lock()->getCurrentPlayer()->getObjects()) {
        if (std::dynamic_pointer_cast<Course::UnitBase>(object) != nullptr)
        {
            std::shared_ptr<Course::UnitBase> unit =
                    std::dynamic_pointer_cast<Course::UnitBase>(object);
            unit->paySalary();;
        }
    }

    //Checks if there's any tiles to be conquered by current player.
    for (auto tile : objectManager_.lock()->getTiles()) {
         tile->conquerTile(playerManager_.lock()->getCurrentPlayer());
    }

    /*Checks if someone has areas that are not connected to the HQ.
     *If there is, they'll become owned by no one. If the current player
     *conquered the headquarters the player will get the opponent's tiles*/
    for (auto player: playerManager_.lock()->getPlayers()) {
        if (player != playerManager_.lock()->getCurrentPlayer()) {

            std::vector<std::shared_ptr<Course::TileBase>> HqConnectedTiles =
                    objectManager_.lock()->getHqConnectedTiles(player);

            //Loops all player's tiles.
            for (auto object : player->getObjects()) {
                if (std::dynamic_pointer_cast<Course::TileBase>(object) != nullptr)
                {
                    std::shared_ptr<Course::TileBase> tile =
                            std::dynamic_pointer_cast<Course::TileBase>(object);

                    /*Tile isn't connected to the HQ and its conquered.
                     *Current player gets the tile in this case*/
                    if (!(std::find(HqConnectedTiles.begin(),
                                    HqConnectedTiles.end(),
                                    tile) != HqConnectedTiles.end())
                                    && HqConnectedTiles.size() == 0)
                    {
                        for (auto unit : tile->getUnits()) {
                            deleteUnitFromTile(unit, tile);
                        }
                        tile->setOwner(playerManager_.lock()->getCurrentPlayer());

                        //Resets farm's crop
                        if (tile->getBuilding() != nullptr
                                && tile->getBuilding()->getType() == "Farm") {
                            std::shared_ptr<Course::Farm> farm =
                                    std::dynamic_pointer_cast
                                    <Course::Farm>(tile->getBuilding());

                            farm->resetFarm();
                        }
                    }

                    /*Tile isn't connected to the HQ and its not conquered.
                     *The tile becomes owned by no one*/
                    else if (!(std::find(HqConnectedTiles.begin(),
                                         HqConnectedTiles.end(),
                                         tile) != HqConnectedTiles.end()))
                    {
                        for (auto unit : tile->getUnits()) {
                            deleteUnitFromTile(unit, tile);
                        }
                        tile->setOwner(nullptr);

                        //Resets farm's crop
                        if (tile->getBuilding() != nullptr
                                && tile->getBuilding()->getType() == "Farm")
                        {
                            std::shared_ptr<Course::Farm> farm =
                                    std::dynamic_pointer_cast
                                    <Course::Farm>(tile->getBuilding());

                            farm->resetFarm();
                        }
                    }
                }
            }

            player->eliminateExcessUnits();
            player->limitResources();
        }
    }

    std::vector<std::shared_ptr<Course::PlayerBase>> lostPlayersThisRound = {};

    /*If any player lost (doesn't own any tiles), the player
     *is marked lost by playermanager and the loss is shown at the menu*/
    for (auto player : playerManager_.lock()->getPlayers()) {
        if (player->getObjects().size() == 0) {
            losingReasons.push_back("conquered");
            playerManager_.lock()->setPlayerAsLost(player,
                                    playerManager_.lock()->getCurrentPlayer());
            lostPlayersThisRound.push_back(player);
        }
    }

    //Player lost due to lack of resources
    for (auto player : playerManager_.lock()->getPlayers()) {
        for (auto res : player->getResources() )
        {
            if (res.second < 0) {
                if (player->getObjects().size() > 0) {
                    losingReasons.push_back("noresources");
                    playerManager_.lock()->setPlayerAsLost(player,
                                      playerManager_.lock()->getCurrentPlayer());
                    lostPlayersThisRound.push_back(player);
                    neutralizePlayer(player);
                    break;
                }
            }
        }
    }

    //Check if any player has 75% of tiles and wins
    for (auto player : playerManager_.lock()->getPlayers()) {
        if (objectManager_.lock()->getTileCountForPlayer(player)
                * 100 / objectManager_.lock()->getTileCount() >= 70)
        {
            for (auto p : playerManager_.lock()->getPlayers()) {
                if (player != p) {
                    playerManager_.lock()->setPlayerAsLost(p);
                    lostPlayersThisRound.push_back(p);
                    neutralizePlayer(p);
                }
            }
        }
    }

    playerManager_.lock()->changeTurn();
    objectManager_.lock()->removeBlockTileOverlays();

    if (playerManager_.lock()->getPlayers().size() == 0) {
        menuObjectManager_.lock()->setTieMenu(lostPlayersThisRound, losingReasons);
    }

    // All the menu interactions for winning and losin
    else if (playerManager_.lock()->getPlayers().size() == 1) {
        // Someone wins
        menuObjectManager_.lock()->setWinMenu
                (playerManager_.lock()->getPlayers().at(0));
    }
    else if (lostPlayersThisRound.size() == 0) {
        // Nobody lost this round (Normal situation)
        openDefaultMenuView();
    }
    else {
        // There is at least one player who lost this round.
        // Possible situations: 1 Player lost, 2 Players lost.
        // If 3 players lose same round it is a win.
        menuObjectManager_.lock()
                ->setPlayerLostMenu(lostPlayersThisRound, losingReasons);
    }

}

void GameEventHandler::neutralizePlayer(std::shared_ptr<Course::PlayerBase> player)
{
    for (auto object : player->getObjects()) {
        if (std::dynamic_pointer_cast<Course::TileBase>(object) != nullptr)
        {
            std::shared_ptr<Course::TileBase> tile =
                    std::dynamic_pointer_cast<Course::TileBase>(object);

            for (auto unit : tile->getUnits()) {
                deleteUnitFromTile(unit, tile);
            }
            tile->setOwner(nullptr);

            if (tile->getBuilding() != nullptr) {
                if (tile->getBuilding()->getType() == "Farm") {
                    std::shared_ptr<Course::Farm> farm =
                            std::dynamic_pointer_cast
                            <Course::Farm>(tile->getBuilding());

                    farm->resetFarm();
                    updateTile(tile);
                }
                if (tile->getBuilding()->getType() == "Headquarters") {
                    std::shared_ptr<Course::HeadQuarters> HQ =
                            std::dynamic_pointer_cast
                            <Course::HeadQuarters>(tile->getBuilding());
                    HQ->setConquered();
                    updateTile(tile);

                }
            }
        }
    }
}


void GameEventHandler::updateAnimatedTileToStatic
                (std::shared_ptr<Course::TileBase> tile, int frame)
{
    if (gameScene_.lock()->isObjectInScene(tile))
    {
        gameScene_.lock()->getObjectInScene
                (tile->getBuilding())->setAnimationFrame(frame);

        gameScene_.lock()->updateTile(tile);
    }
}


void GameEventHandler::updateForest(std::string status,
                           std::shared_ptr<Course::TileBase> tile,
                           const std::shared_ptr<Course::BuildingBase>& building)
{
    if (gameScene_.lock()->isObjectInScene(tile)) {
        if (status == "Cut") {
            tile->setImageFiles(ImageVectors::FOREST_STUMPS);
        }
        else if (status == "Grow"){
            srand (time(NULL));
            int randomNum = rand() % 2 + 1;
            if (randomNum == 1) {
                tile->setImageFiles(ImageVectors::FOREST_1);
            }
            else if (randomNum == 2) {
                tile->setImageFiles(ImageVectors::FOREST_2);
            }
        }
        gameScene_.lock()->updateItem(tile);
        if (status == "Grassland") {

            std::shared_ptr<Course::Grassland> newTile =
                    std::make_shared<Course::Grassland>(
                                    tile->getCoordinate(),
                                    1, 1,
                                    shared_from_this(),
                                    objectManager_);
            playerManager_.lock()->getCurrentPlayer()->removeObject(tile);
            newTile->setGameSettings(gameSettingsManager_.lock());
            objectManager_.lock()->replaceTile(tile, newTile);

            newTile->setImageFiles(ImageVectors::GRASSLAND);

            gameScene_.lock()->removeItem(tile);
            gameScene_.lock()->drawItem(newTile);
            newTile->addBuilding(building);
            updateTile(newTile);
            setTileInspectionMenuView(newTile);
        }
    }
}


void GameEventHandler::setTileInspectionMenuView(
        std::shared_ptr<Course::TileBase> tile, int index_for_buildings)
{
    if (unitToDeploy_ != nullptr) return;
    menuObjectManager_.lock()
            ->setTileInspectionMenuView(tile, index_for_buildings);

}


void GameEventHandler::openStatsMenuView()
{
    menuObjectManager_.lock()->setStatMenuView();
}


void GameEventHandler::openDefaultMenuView()
{
    unitToDeploy_ = nullptr;
    unitPreviousTile_ = nullptr;
    gameScene_.lock()->removeMouseFollowItem();
    menuObjectManager_.lock()->setDefaultMenuView();
    objectManager_.lock()->removeClickedTileBorder();
    objectManager_.lock()->removeBlockTileOverlays();
}


void GameEventHandler::openUnitBuyMenu() {
    menuObjectManager_.lock()->setUnitShopMenuView();
    objectManager_.lock()->removeClickedTileBorder();
    objectManager_.lock()->removeBlockTileOverlays();
}


void GameEventHandler::createUnit(std::string unit)
{
    if (unitToDeploy_ != nullptr) {
        unitToDeploy_ = nullptr;
        unitPreviousTile_ = nullptr;
        gameScene_.lock()->removeMouseFollowItem();
        objectManager_.lock()->removeBlockTileOverlays();
        return;
    }
    std::shared_ptr<Course::UnitBase> unitToPlace = nullptr;

    if (unit == "BasicWorker") {
        unitToPlace = std::make_shared<Course::BasicWorker>(
                    shared_from_this(),
                    objectManager_,
                    gameSettingsManager_,
                    playerManager_.lock()->getCurrentPlayer()
                    );
        unitToPlace->setImageFiles(ImageVectors::BASICWORKER);
    }
    if (unit == "Expert") {
        unitToPlace = std::make_shared<Expert>(
                    shared_from_this(),
                    objectManager_,
                    gameSettingsManager_,
                    playerManager_.lock()->getCurrentPlayer()
                    );
        unitToPlace->setImageFiles(ImageVectors::EXPERT);
    }
    if (unit == "Soldier") {
        unitToPlace = std::make_shared<Soldier>(
                    shared_from_this(),
                    objectManager_,
                    gameSettingsManager_,
                    playerManager_.lock()->getCurrentPlayer()
                    );
        unitToPlace->setImageFiles(ImageVectors::SOLDIER);

    }

    unitToPlace->setAnimationOption(AnimationOptions::UNIT);

    if (unit == "Soldier" && playerManager_.lock()
            ->getCurrentPlayer()->getFreeSoldierAmount() <= 0) return;
    if ((unit == "BasicWorker" or unit == "Expert") and
            playerManager_.lock()->getCurrentPlayer()
            ->getFreeUnitAmount() <= 0) return;
    if (!canBuyUnitOrBuilding(unitToPlace)) return;

    unitToDeploy_ = unitToPlace;
    objectManager_.lock()->addBlockTileOverlays();

    gameScene_.lock()->addMouseFollowPicture(unitToPlace->getImageFiles());
}


void GameEventHandler::moveUnitFromTile(int index,
                                 std::shared_ptr<Course::TileBase> tile)
{
    if (unitToDeploy_ != nullptr or unitPreviousTile_ != nullptr) {
        unitPreviousTile_ = nullptr;
        unitToDeploy_ = nullptr;
        gameScene_.lock()->removeMouseFollowItem();
        objectManager_.lock()->removeBlockTileOverlays();
        return;

    }

    unitPreviousTile_ = tile;
    if (tile->getOwner() == playerManager_.lock()->getCurrentPlayer()) {
        unitToDeploy_ = tile->getUnits().at(index);
    } else {
        unitToDeploy_ = tile->getConqueringUnits().at(index);
    }

    objectManager_.lock()->addBlockTileOverlays();
    if (unitToDeploy_->getType() == "BasicWorker") {
        gameScene_.lock()->addMouseFollowPicture(ImageVectors::BASICWORKER);
    }
    if (unitToDeploy_->getType() == "Expert") {
        gameScene_.lock()->addMouseFollowPicture(ImageVectors::EXPERT);
    }
    if (unitToDeploy_->getType() == "Soldier") {
        gameScene_.lock()->addMouseFollowPicture(ImageVectors::SOLDIER);
    }

}


void GameEventHandler::buyUnitOrBuilding(
    std::shared_ptr<Course::PlaceableGameObject> object) {

    Course::ResourceMap cost = object->getCost();

    if (!playerManager_.lock()->getCurrentPlayer()->hasEnoughResources(cost)) {
        qDebug()<<"Not enough resources!";
        return;
    }

    playerManager_.lock()->getCurrentPlayer()->addOrRemoveResources(cost);
}


bool GameEventHandler::canBuyUnitOrBuilding(
          std::shared_ptr<Course::PlaceableGameObject> object)
{
    Course::ResourceMap cost = object->getCost();
    if (!playerManager_.lock()->getCurrentPlayer()->hasEnoughResources(cost)) {
        qDebug()<<"Not enough resources!";
        return false;
    }
    return true;
}


void GameEventHandler::deleteUnitFromTile(int index,
                                     std::shared_ptr<Course::TileBase> tile)
{
    if (unitToDeploy_ != nullptr or unitPreviousTile_ != nullptr) return;

    if (tile->getOwner() == playerManager_.lock()->getCurrentPlayer()) {
        gameScene_.lock()->removeItem(tile->getUnits().at(index));
        tile->removeUnit(tile->getUnits().at(index));
    } else {
        gameScene_.lock()->removeItem(tile->getConqueringUnits().at(index));
        tile->removeUnit(tile->getConqueringUnits().at(index));
    }

    objectManager_.lock()->removeClickedTileBorder();
    setTileInspectionMenuView(tile);
    tile->updateAnimation();
}


void GameEventHandler::deleteUnitFromTile(std::shared_ptr<Course::UnitBase> unit,
                                          std::shared_ptr<Course::TileBase> tile)
{
    gameScene_.lock()->removeItem(unit);
    tile->removeUnit(unit);
}


void GameEventHandler::updateTile(std::shared_ptr<Course::TileBase> tile)
{
    gameScene_.lock()->updateTile(tile);
}


void GameEventHandler::buildBuilding(std::string building_string,
                                     std::shared_ptr<Course::TileBase> tile)
{
    if (unitPreviousTile_ != nullptr or unitToDeploy_ != nullptr) return;

    std::shared_ptr<Course::BuildingBase> building;
    if (building_string == "Village") {
         building = std::make_shared<Student::Village>
                (shared_from_this(),
                 objectManager_,
                 playerManager_.lock()->getCurrentPlayer());

        building->setImageFiles(ImageVectors::VILLAGE);
        building->setAnimationOption(AnimationOptions::EMPTY);

    }
    if (building_string == "Outpost") {
         building = std::make_shared<Course::Outpost>
                (shared_from_this(),
                 objectManager_,
                 playerManager_.lock()->getCurrentPlayer());

        building->setImageFiles(ImageVectors::OUTPOST);
        building->setAnimationOption(AnimationOptions::OUTPOST);

    }
    if (building_string == "Nuclear Power Plant") {
         building = std::make_shared<Student::NuclearPlant>
                (shared_from_this(),
                 objectManager_,
                 playerManager_.lock()->getCurrentPlayer());

        building->setImageFiles(ImageVectors::NUCLEARPLANT);
        building->setAnimationOption(AnimationOptions::NUCLEAR);

    }
    if (building_string == "Mine") {
         building = std::make_shared<Student::Mine>
                (shared_from_this(),
                 objectManager_,
                 playerManager_.lock()->getCurrentPlayer());

        building->setImageFiles(ImageVectors::MINE);
        building->setAnimationOption(AnimationOptions::EMPTY);

    }
    if (building_string == "Hydroelectric Power Plant") {
         building = std::make_shared<Student::HydroPower>
                (shared_from_this(),
                 objectManager_,
                 playerManager_.lock()->getCurrentPlayer());

        if (std::dynamic_pointer_cast<Student::River>(tile) != nullptr) {
            int orientation = std::dynamic_pointer_cast
                            <Student::River>(tile)->getRiverOrientation();
            if (orientation == 1) {
                building->setImageFiles(ImageVectors::HYDROPOWERNS);
            }
            if (orientation == 0) {
                building->setImageFiles(ImageVectors::HYDROPOWERWE);
            }
        }

        building->setAnimationOption(AnimationOptions::HEPP);

    }
    if (building_string == "Farm") {
         building = std::make_shared<Course::Farm>
                (shared_from_this(),
                 objectManager_,
                 playerManager_.lock()->getCurrentPlayer());

        building->setImageFiles(ImageVectors::FARM);
        building->setAnimationOption(AnimationOptions::EMPTY);

    }
    if (building_string == "Bridge") {
         building = std::make_shared<Student::Bridge>
                (shared_from_this(),
                 objectManager_,
                 playerManager_.lock()->getCurrentPlayer());

         if (std::dynamic_pointer_cast<Student::River>(tile) != nullptr) {
             int orientation = std::dynamic_pointer_cast<Student::River>(tile)->getRiverOrientation();
             if (orientation == 1) {
                 building->setImageFiles(ImageVectors::BRIDGEWE);
             }
             if (orientation == 0) {
                 building->setImageFiles(ImageVectors::BRIDGENS);
             }
         }
        building->setAnimationOption(AnimationOptions::EMPTY);

    }

    if (!canBuyUnitOrBuilding(building)) return;

    buyUnitOrBuilding(building);

    tile->addBuilding(building);
    updateTile(tile);
    setTileInspectionMenuView(tile);
    tile->updateAnimation();
}


Course::ResourceMap GameEventHandler::getCurrentRevenue() {
    Course::ResourceMap revenue = Course::ConstResourceMaps::NO_RESOURCES;
    for (auto tile : objectManager_.lock()->getTiles()) {
        if (tile->getOwner() == playerManager_.lock()->getCurrentPlayer()) {
            Course::ResourceMap r;
            r = tile->getCurrentRevenue();
            revenue = Course::mergeResourceMaps(revenue, r);
        }
    }
    return revenue;
}


Course::ResourceMap GameEventHandler::getCurrentExpences() {
    Course::ResourceMap expenses = {};
    for (auto tile : objectManager_.lock()->getTiles()) {
        if (tile->getOwner() == playerManager_.lock()->getCurrentPlayer()) {
            expenses = mergeResourceMaps(expenses, tile->getCurrentExpenses());
        }
        for (auto unit : tile->getConqueringUnits()) {
            if (unit->getOwner() == playerManager_.lock()->getCurrentPlayer()) {
                expenses = mergeResourceMaps(expenses, unit->getSalary());
            }
        }
    }
    return expenses;
}


Course::ResourceMap GameEventHandler::getCurrentNet() {
    Course::ResourceMap net = {};
    for (auto tile : objectManager_.lock()->getTiles()) {
        if (tile->getOwner() == playerManager_.lock()->getCurrentPlayer()) {
            net = mergeResourceMaps(net, tile->getCurrentNet());
        }
        for (auto unit : tile->getConqueringUnits()) {
            if (unit->getOwner() == playerManager_.lock()->getCurrentPlayer()) {
                net = mergeResourceMaps(net, unit->getSalary());
            }
        }
    }
    return net;
}


std::shared_ptr<Course::PlayerBase> GameEventHandler::getCurrentPlayer()
{
    return playerManager_.lock()->getCurrentPlayer();
}


void GameEventHandler::restartGame()
{
    emit restartGameSignal();
}


} //Namespace Student

