pyebm is a package that aims to collect together code for doing learning and inference in energy-based models, including but not limited to restricted Boltzmann machines, and related models such as deep belief nets and deep neural networks/autoencoders pretrained in an unsupervised fashion.

From the abstract for [Yann LeCun's NIPS 2006 talk](http://nips.cc/Conferences/2006/Program/event.php?ID=3):

> "Energy-Based Models (EBM) capture dependencies between variables by associating a scalar energy to each configuration of the variables. Given a set of observed variables, an EBM inference consists in finding configurations of unobserved variables that minimize the energy. Training an EBM consists in designing a loss function whose minimization will shape the energy surface so that desired variable configurations have lower energies than undesired configurations."

pyebm uses the [NumPy](http://nips.cc/Conferences/2006/Program/event.php?ID=3) numerical library for Python, and parts may depend on [matplotlib](http://matplotlib.sourceforge.net/) plotting library for various visualizations.