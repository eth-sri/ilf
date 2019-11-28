var crowdsale = artifacts.require("Crowdsale");

module.exports = function(deployer) {
  deployer.deploy(crowdsale);
};
