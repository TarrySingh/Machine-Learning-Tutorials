/**
 * A BitSet similar to java.util.BitSet.
 *
 * <p>JavaScript Note: There is no good way to implement something like this in 
 * JavaScript.  JS has no true int type, arrays are usually implemented as
 * hashes, etc.  This class should probably be nixed for something that is
 * similarly (in)efficient, but more clear.</p>
 *
 * @class
 * @param {Number|Array} [bits] a 32 bit number or array of 32 bit numbers
 *                              representing the bitset.  These are typically
 *                              generated by the ANTLR Tool.
 */
org.antlr.runtime.BitSet = function(bits) {
    if (!bits) {
        bits = org.antlr.runtime.BitSet.BITS;
    }

    if (org.antlr.lang.isArray(bits)) {
        /**
         * An array of Numbers representing the BitSet.
         * @type Array
         */
        this.bits = bits;
    } else if(org.antlr.lang.isNumber(bits)) {
        this.bits = [];
    }
};

org.antlr.lang.augmentObject(org.antlr.runtime.BitSet, {
    /**
     * Number of bits in each number.
     * @constant
     * @memberOf org.antlr.runtime.BitSet
     */
    BITS: 32,

    /**
     * Log (base 2) of the number of bits in each number.
     * @constant
     * @memberOf org.antlr.runtime.BitSet
     */
    LOG_BITS: 5,  // 2^5 == 32 

    /**
     * We will often need to do a mod operator (i mod nbits).  Its
     * turns out that, for powers of two, this mod operation is
     * same as (i & (nbits-1)).  Since mod is slow, we use a
     * precomputed mod mask to do the mod instead.
     * @constant
     * @memberOf org.antlr.runtime.BitSet
     */
    MOD_MASK: 31, // BITS - 1

    /**
     * Create mask for bit modded to fit in a single word.
     * @example
     * bitmask(35) => 00000000000000000000000000000100
     * bitmask(3)  => 00000000000000000000000000000100
     * @param {Number} bitNumber the bit to create a mask for.
     * @returns {Number} the bitmask.
     * @memberOf org.antlr.runtime.BitSet
     * @private
     */
    bitMask: function(bitNumber) {
        var bitPosition = bitNumber & org.antlr.runtime.BitSet.MOD_MASK;
        return 1 << bitPosition;
    },

    /**
     * Calculate the minimum number of bits needed to represent el.
     * @param {Number} el a number to be included in the BitSet.
     * @returns {Number} the number of bits need to create a BitSet with member
     *                   el.
     * @memberOf org.antlr.runtime.BitSet
     * @private
     */
    numWordsToHold: function(el) {
        return (el >> org.antlr.runtime.BitSet.LOG_BITS) + 1;
    },

    /**
     * @param {Number} bit a number to be included in the BitSet
     * @returns {Number} the index of the word in the field bits that would
     *                   hold bit.
     * @memberOf org.antlr.runtime.BitSet
     * @private
     */
    wordNumber: function(bit) {
        return bit >> org.antlr.runtime.BitSet.LOG_BITS; // bit / BITS
    },

    /**
     * BitSet factory method.
     * 
     * <p>Operates in a number of modes:
     * <ul>
     * <li>If el is a number create the BitSet containing that number.</li>
     * <li>If el is an array create the BitSet containing each number in the
     * array.</li>
     * <li>If el is a BitSet return el.</li>
     * <li>If el is an Object create the BitSet containing each numeric value
     * in el.</li>
     * <li>If el is a number and el2 is a number return a BitSet containing
     * the numbers between el and el2 (inclusive).</li>
     * </ul>
     * </p>
     * @param {Number|Array|org.antlr.runtime.BitSet|Object} el
     * @param {Number} el2
     * @returns {org.antlr.runtime.BitSet}
     * @memberOf org.antlr.runtime.BitSet
     */
    of: function(el, el2) {
        var i, n, s, keys;

        if (org.antlr.lang.isNumber(el)) {
            if (org.antlr.lang.isNumber(el2)) {
                s = new org.antlr.runtime.BitSet(el2 + 1);
                for (i = el; i <= el2; i++) {
                    n = org.antlr.runtime.BitSet.wordNumber(i);
                    s.bits[n] |= org.antlr.runtime.BitSet.bitMask(i);
                }
                return s;
            } else {
                s = new org.antlr.runtime.BitSet(el + 1);
                s.add(el);
                return s;
            }
        } else if(org.antlr.lang.isArray(el)) {
            s = new org.antlr.runtime.BitSet();
            for (i=el.length-1; i>=0; i--) {
                s.add(el[i]);
            }
            return s;
        } else if (el instanceof org.antlr.runtime.BitSet) {
            if (!el) {
                return null;
            }
            return el;
        } else if (el instanceof org.antlr.runtime.IntervalSet) {
            if (!el) {
                return null;
            }
            s = new org.antlr.runtime.BitSet();
            s.addAll(el);
            return s;
        } else if (org.antlr.lang.isObject(el)) {
            keys = [];
            for (i in el) {
                if (org.antlr.lang.isNumber(i)) {
                    keys.push(i);
                }
            }
            return org.antlr.runtime.BitSet.of(keys);
        }
    }
});



org.antlr.runtime.BitSet.prototype = {
    /**
     * Add el into this set.
     * @param {Number} el the number to add to the set.
     */
    add: function(el) {
        var n = org.antlr.runtime.BitSet.wordNumber(el);
        if (n >= this.bits.length) {
            this.growToInclude(el);
        }
        this.bits[n] |= org.antlr.runtime.BitSet.bitMask(el);
    },

    /**
     * Add multiple elements into this set.
     * @param {Array|org.antlr.runtime.BitSet} elements the elements to be added to
     *                                           this set.
     */
    addAll: function(elements) {
        var other,
            i,
            e;

        if ( elements instanceof org.antlr.runtime.BitSet ) {
            this.orInPlace(elements);
        }
		else if ( elements instanceof org.antlr.runtime.IntervalSet ) {
			other = elements;
			// walk set and add each interval
            /* @todo after implementing intervalset
			for (Iterator iter = other.intervals.iterator(); iter.hasNext();) {
				Interval I = (Interval) iter.next();
				this.orInPlace(BitSet.range(I.a,I.b));
			}*/
		} else if (org.antlr.lang.isArray(elements)) {
    		for (i = 0; i < elements.length; i++) {
	    		e = elements[i];
		    	this.add(e);
    		}
        } else {
            return;
        }
	},

    /**
     * Clone this BitSet and then {@link #andInPlace} with a.
     * @param {org.antlr.runtime.BitSet} a a bit set.
     * @returns {org.antlr.runtime.BitSet}
     */
    and: function(a) {
        var s = this.clone();
        s.andInPlace(a);
        return s;
    },

    /**
     * Perform a logical AND of this target BitSet with the argument BitSet.
     *
     * This bit set is modified so that each bit in it has the value true if 
     * and only if it both initially had the value true and the corresponding 
     * bit in the bit set argument also had the value true. 
     * @param {org.antlr.runtime.BitSet} a a bit set.
     * @returns {org.antlr.runtime.BitSet}
     */
    andInPlace: function(a) {
        var min = Math.min(this.bits.length, a.bits.length),
            i;
        for (i = min - 1; i >= 0; i--) {
            this.bits[i] &= a.bits[i];
        }
        // clear all bits in this not present in a (if this bigger than a).
        for (i = min; i < this.bits.length; i++) {
            this.bits[i] = 0;
        }
    },

    /**
     * Clear all bits or a specific bit.
     *
     * If no arguments given, sets all of the bits in this BitSet to false.
     * If one argument given, sets the bit specified by the index to false.
     * @param {Number} [el] the index of the bit to be cleared.
     */
    clear: function(el) {
        if (arguments.length===0) {
            var i;
            for (i = this.bits.length - 1; i >= 0; i--) {
                this.bits[i] = 0;
            }
            return;
        }

        var n = org.antlr.runtime.BitSet.wordNumber(el);
        if (n >= this.bits.length) {	// grow as necessary to accommodate
            this.growToInclude(el);
        }
        this.bits[n] &= ~org.antlr.runtime.BitSet.bitMask(el);
    },

    /**
     * Cloning this BitSet produces a new BitSet  that is equal to it. 
     *
     * The clone of the bit set is another bit set that has exactly the same
     * bit set to true as this bit set. 
     * @returns {org.antlr.runtime.BitSet} a clone of this BitSet.
     */
    clone: function() {
        var i, len, b=[];
        for (i=0, len=this.bits.length; i<len; i++) {
            b[i] = this.bits[i];
        }
        return new org.antlr.runtime.BitSet(b);
    },

    /**
     * Returns the number of bits of space actually in use by this BitSet to 
     * represent bit values.
     *
     * The maximum element in the set is the size - 1st element. 
     * @returns {Number} the number of bits currently in this bit set.
     */
    size: function() {
        var deg = 0, i, word, bit;
        for (i = this.bits.length - 1; i >= 0; i--) {
            word = this.bits[i];
            if (word !== 0) {
                for (bit = org.antlr.runtime.BitSet.BITS - 1; bit >= 0; bit--) {
                    if ((word & (1 << bit)) !== 0) {
                        deg++;
                    }
                }
            }
        }
        return deg;
    },

    /**
     * Compares this object against the specified object.
     *
     * The result is true if and only if the argument is not null and is a
     * BitSet object that has exactly the same set of bits set to true as
     * this bit set. That is, for every nonnegative int index k,
     * <pre><code>
     * ((BitSet)obj).get(k) == this.get(k)
     * </code></pre>
     * must be true. The current sizes of the two bit sets are not compared.
     * @param {Object} other the object to compare with.
     * @returns {Boolean} if the objects are the same; false otherwise.
     */
    equals: function(other) {
        if ( !other || !(other instanceof org.antlr.runtime.BitSet) ) {
            return false;
        }

        var otherSet = other,
            i,
            n = Math.min(this.bits.length, otherSet.bits.length);

        // for any bits in common, compare
        for (i=0; i<n; i++) {
            if (this.bits[i] != otherSet.bits[i]) {
                return false;
            }
        }

        // make sure any extra bits are off

        if (this.bits.length > n) {
            for (i = n+1; i<this.bits.length; i++) {
                if (this.bits[i] !== 0) {
                    return false;
                }
            }
        }
        else if (otherSet.bits.length > n) {
            for (i = n+1; i<otherSet.bits.length; i++) {
                if (otherSet.bits[i] !== 0) {
                    return false;
                }
            }
        }

        return true;
    },

    /**
     * Grows the set to a larger number of bits.
     * @param {Number} bit element that must fit in set
     * @private
     */
    growToInclude: function(bit) {
        var newSize = Math.max(this.bits.length << 1, org.antlr.runtime.BitSet.numWordsToHold(bit)),
            newbits = [], //new Array(newSize),
            i;
        for (i=0, len=this.bits.length; i<len; i++) {
            newbits[i] = this.bits[i];
        }
        this.bits = newbits;
    },

    /**
     * Returns the value of the bit with the specified index.
     *
     * The value is true if the bit with the index el is currently set 
     * in this BitSet; otherwise, the result is false.
     * @param {Number} el the bit index.
     * @returns {Boolean} the value of the bit with the specified index.
     */
    member: function(el) {
        var n = org.antlr.runtime.BitSet.wordNumber(el);
        if (n >= this.bits.length) { return false; }
        return (this.bits[n] & org.antlr.runtime.BitSet.bitMask(el)) !== 0;
    },

    /**
     * Returns the index of the first bit that is set to true.
     * If no such bit exists then -1 is returned.
     * @returns {Number} the index of the next set bit.
     */
    getSingleElement: function() {
        var i;
        for (i = 0; i < (this.bits.length << org.antlr.runtime.BitSet.LOG_BITS); i++) {
            if (this.member(i)) {
                return i;
            }
        }
        return -1; //Label.INVALID;
    },

    /**
     * Returns true if this BitSet contains no bits that are set to true.
     * @returns {Boolean} boolean indicating whether this BitSet is empty.
     */
    isNil: function() {
        var i;
        for (i = this.bits.length - 1; i >= 0; i--) {
            if (this.bits[i] !== 0) {
                return false;
            }
        }
        return true;
    },

    /**
     * If a bit set argument is passed performs a {@link #subtract} of this bit
     * set with the argument bit set.  If no argument is passed, clone this bit
     * set and {@link #notInPlace}.
     * @param {org.antlr.runtime.BitSet} [set]
     * @returns {org.antlr.runtime.BitSet}
     */
    complement: function(set) {
        if (set) {
            return set.subtract(this);
        } else {
            var s = this.clone();
            s.notInPlace();
            return s;
        }
    },

    /**
     * If no arguments are passed sets all bits to the complement of their
     * current values.  If one argument is passed sets each bit from the
     * beginning of the bit set to index1 (inclusive) to the complement of its
     * current value.  If two arguments are passed sets each bit from the
     * specified index1 (inclusive) to the specified index2 (inclusive) to the
     * complement of its current value.
     * @param {Number} index1
     * @param {Number} index2
     */
    notInPlace: function() {
        var minBit, maxBit, i, n;
        if (arguments.length===0) {
            for (i = this.bits.length - 1; i >= 0; i--) {
                this.bits[i] = ~this.bits[i];
            }
        } else {
            if (arguments.length===1) {
                minBit = 0;
                maxBit = arguments[0];
            } else {
                minBit = arguments[0];
                maxBit = arguments[1];
            }
            // make sure that we have room for maxBit
            this.growToInclude(maxBit);
            for (i = minBit; i <= maxBit; i++) {
                n = org.antlr.runtime.BitSet.wordNumber(i);
                this.bits[n] ^= org.antlr.runtime.BitSet.bitMask(i);
            }
        }

    },

    /**
     * Performs a logical OR of this bit set with the bit set argument.
     * If no argument is passed, return this bit set.  Otherwise a clone of
     * this bit set is modified so that a bit in it has the value true if and
     * only if it either already had the value true or the corresponding bit
     * in the bit set argument has the value true.
     * @param {org.antlr.runtime.BitSet} [a] a bit set.
     * @returns {org.antlr.runtime.BitSet}
     */
    or: function(a) {
		if ( !a ) {
			return this;
		}
        var s = this.clone();
        s.orInPlace(a);
        return s;
    },

    /**
     * Performs a logical {@link #or} in place.
     * @param {org.antlr.runtime.BitSet} [a]
     * @returns {org.antlr.runtime.BitSet}
     */
    orInPlace: function(a) {
		if ( !a ) {
			return;
		}
        // If this is smaller than a, grow this first
        if (a.bits.length > this.bits.length) {
            this.setSize(a.bits.length);
        }
        var min = Math.min(this.bits.length, a.bits.length),
            i;
        for (i = min - 1; i >= 0; i--) {
            this.bits[i] |= a.bits[i];
        }
    },

    /**
     * Sets the bit specified by the index to false.
     * @param {Number} bitIndex the index of the bit to be cleared.
     */
    remove: function(el) {
        var n = org.antlr.runtime.BitSet.wordNumber(el);
        if (n >= this.bits.length) {
            this.growToInclude(el);
        }
        this.bits[n] &= ~org.antlr.runtime.BitSet.bitMask(el);
    },

    /**
     * Grows the internal bits array to include at least nwords numbers.
     * @private
     * @param {Number} nwords how many words the new set should be
     * @private
     */
    setSize: function(nwords) {
        var n = nwords - this.bits.length;
        while (n>=0) {
            this.bits.push(0);
            n--;
        }
    },

    /**
     * Returns the number of bits capable of being represented by this bit set
     * given its current size.
     * @returns {Number} the maximum number of bits that can be represented at
     *                   the moment.
     * @private
     */
    numBits: function() {
        return this.bits.length << org.antlr.runtime.BitSet.LOG_BITS; // num words * bits per word
    },

    /**
     * Return how much space is being used by the bits array not
     * how many actually have member bits on.
     * @returns {Number} the length of the internal bits array.
     * @private
     */
    lengthInLongWords: function() {
        return this.bits.length;
    },

    /**
     * Is this bit set contained within a?
     * @param {org.antlr.runtime.BitSet} a bit set
     * @returns {Boolean} true if and only if a is a subset of this bit set.
     */
    subset: function(a) {
        if (!a) { return false; }
        return this.and(a).equals(this);
    },

    /**
     * Subtract the elements of the argument bit set from this bit set in place.
     * That is, for each set bit in the argument bit set, set the corresponding
     * bit in this bit set to false.
     * @param {org.antlr.runtime.BitSet} a bit set.
     */
    subtractInPlace: function(a) {
        if (!a) { return; }
        // for all words of 'a', turn off corresponding bits of 'this'
        var i;
        for (i = 0; i < this.bits.length && i < a.bits.length; i++) {
            this.bits[i] &= ~a.bits[i];
        }
    },

    /**
     * Perform a {@link #subtractInPlace} on a clone of this bit set.
     * @param {org.antlr.runtime.BitSet} a bit set.
     * @returns {org.antlr.runtime.BitSet} the new bit set.
     */
    subtract: function(a) {
        if (!a || !(a instanceof org.antlr.runtime.BitSet)) { return null; }

        var s = this.clone();
        s.subtractInPlace(a);
        return s;
    },

    /* antlr-java needs this to make its class hierarchy happy . . .
    toList: function() {
		throw new Error("BitSet.toList() unimplemented");
	},
    */

    /**
     * Creates an array of the indexes of each bit set in this bit set.
     * @returns {Array}
     */
    toArray: function() {
        var elems = [], //new Array(this.size()),
            i,
            en = 0;
        for (i = 0; i < (this.bits.length << org.antlr.runtime.BitSet.LOG_BITS); i++) {
            if (this.member(i)) {
                elems[en++] = i;
            }
        }
        return elems;
    },

    /**
     * Returns the internal representation of this bit set.
     * This representation is an array of numbers, each representing 32 bits.
     * @returns {Array}
     */
    toPackedArray: function() {
        return this.bits;
    },

    /**
     * Returns a string representation of this bit set.
     * <p>For every index for which this BitSet contains a bit in the set state,
     * the decimal representation of that index is included in the result.
     * Such indices are listed in order from lowest to highest, separated by
     * ", " (a comma and a space) and surrounded by braces, resulting in the
     * usual mathematical notation for a set of integers.</p>
     * 
     * <p>If a grammar g is passed, print g.getTokenDisplayName(i) for each set
     * index instead of the numerical index.</p>
     *
     * <>If two arguments are passed, the first will be used as a custom
     * separator string.  The second argument is an array whose i-th element
     * will be added if the corresponding bit is set.</p>
     *
     * @param {Object|String} [arg1] an Object with function property
     *      getTokenDispalyName or a String that will be used as a list
     *      separator.
     * @param {Array} [vocabulary] array from which the i-th value will be
     *      drawn if the corresponding bit is set.  Must pass a string as the
     *      first argument if using this option.
     * @return A commma-separated list of values
     */
    toString: function() {
        if (arguments.length===0) {
            return this.toString1(null);
        } else {
            if (org.antlr.lang.isString(arguments[0])) {
                if (!org.antlr.lang.isValue(arguments[1])) {
                    return this.toString1(null);
                } else {
                    return this.toString2(arguments[0], arguments[1]);
                }
            } else {
                return this.toString1(arguments[0]);
            }
        }
    },

    /**
     * Transform a bit set into a string by formatting each element as an
     * integer separator The string to put in between elements
     * @private
     * @return A commma-separated list of values
     */
    toString1: function(g) {
        var buf = "{",
            separator = ",",
            i,
		    havePrintedAnElement = false;

        for (i = 0; i < (this.bits.length << org.antlr.runtime.BitSet.LOG_BITS); i++) {
            if (this.member(i)) {
                if (i > 0 && havePrintedAnElement ) {
                    buf += separator;
                }
                if ( g ) {
                    buf += g.getTokenDisplayName(i);
                }
                else {
                    buf += i.toString();
                }
				havePrintedAnElement = true;
            }
        }
        return buf + "}";
    },

    /**
     * Create a string representation where instead of integer elements, the
     * ith element of vocabulary is displayed instead.  Vocabulary is a Vector
     * of Strings.
     * separator The string to put in between elements
     * @private
     * @return A commma-separated list of character constants.
     */
    toString2: function(separator, vocabulary) {
        var str = "",
            i;
        for (i = 0; i < (this.bits.length << org.antlr.runtime.BitSet.LOG_BITS); i++) {
            if (this.member(i)) {
                if (str.length > 0) {
                    str += separator;
                }
                if (i >= vocabulary.size()) {
                    str += "'" + i + "'";
                }
                else if (!org.antlr.lang.isValue(vocabulary.get(i))) {
                    str += "'" + i + "'";
                }
                else {
                    str += vocabulary.get(i);
                }
            }
        }
        return str;
    }
};
