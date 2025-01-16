package io.github.ericmedvet.jsdynsym.control;

import io.github.ericmedvet.jnb.datastructure.Pair;
import io.github.ericmedvet.jsdynsym.core.DynamicalSystem;

public interface BiEnvironment<O1, O2, A1, A2, S> extends DynamicalSystem<Pair<A1,A2>, Pair<O1,O2>, S> {

    Pair<A1, A2> defaultActions();
}
