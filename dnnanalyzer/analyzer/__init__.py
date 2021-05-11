analyzers = {

}


def create_analyzer(name, model, **kwargs):
    """
    Instantiates the analyzer with the name 'name'
    This convenience function takes an analyzer name
    creates the respective analyzer.
    Alternatively analyzers can be created directly by
    instantiating the respective classes.
    :param name: Name of the analyzer.
    :param model: The model to analyze, passed to the analyzer's __init__.
    :param kwargs: Additional parameters for the analyzer's .
    :return: An instance of the chosen analyzer.
    :raise KeyError: If there is no analyzer with the passed name.
    """
    try:
        analyzer_class = analyzers[name]
    except KeyError:
        raise KeyError(
            "No analyzer with the name '%s' could be found."
            " All possible names are: %s" % (name, list(analyzers.keys())))
    return analyzer_class(model, **kwargs)
