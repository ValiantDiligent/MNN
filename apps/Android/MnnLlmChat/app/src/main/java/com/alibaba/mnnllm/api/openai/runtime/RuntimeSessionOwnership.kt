package com.alibaba.mnnllm.api.openai.runtime

import java.util.Collections
import java.util.IdentityHashMap

/** An independent identity lease, including when acquiring an already loaded session. */
class SessionLeaseToken internal constructor()

/** All mutations run under the controller lock; only the last lease releases the native session. */
internal class RuntimeSessionOwnership<T : Any> {
    var session: T? = null
        private set
    private val leases = Collections.newSetFromMap(IdentityHashMap<SessionLeaseToken, Boolean>())

    fun acquire(session: T): SessionLeaseToken {
        check(this.session == null || this.session === session) { "Release the previous session before replacing it" }
        this.session = session
        return SessionLeaseToken().also { leases.add(it) }
    }

    /** Returns true when a lease was removed (or an administrative force release was performed). */
    fun release(expected: SessionLeaseToken? = null, release: (T) -> Unit): Boolean {
        val owned = session ?: return false
        if (expected != null) {
            if (!leases.remove(expected)) return false
            if (leases.isNotEmpty()) return true
        }
        // null is an explicit administrative force release, invalidating every outstanding lease.
        session = null
        leases.clear()
        release(owned)
        return true
    }
}
