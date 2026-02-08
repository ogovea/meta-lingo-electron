/**
 * CQL String to BuilderElement[] Parser
 * Converts a CQL query string back into visual builder elements
 * for template restoration and CQL import
 */

import type {
  BuilderElement,
  TokenAttribute,
  ComparisonOperator,
  ConditionGroup,
  TokenCondition
} from './types'
import { generateId } from './constants'

/**
 * Parse a CQL string into an array of BuilderElements
 * Handles: [attr="val"], [], []{n,m}, []?, []*, "word", |
 */
export function parseCQLToElements(cql: string): BuilderElement[] {
  const elements: BuilderElement[] = []
  let pos = 0
  const query = cql.trim()

  while (pos < query.length) {
    // Skip whitespace
    while (pos < query.length && query[pos] === ' ') pos++
    if (pos >= query.length) break

    if (query[pos] === '[') {
      const result = parseBracketToken(query, pos)
      elements.push(result.element)
      pos = result.endPos
    } else if (query[pos] === '"') {
      // Parse quoted word
      const endQuote = query.indexOf('"', pos + 1)
      if (endQuote === -1) break
      const word = query.slice(pos + 1, endQuote)
      elements.push({
        id: generateId(),
        type: 'normal_token',
        conditionGroups: [{
          conditions: [{
            id: generateId(),
            attribute: 'word',
            operator: '=',
            value: word
          }],
          logic: 'and'
        }]
      })
      pos = endQuote + 1
    } else if (query[pos] === '|') {
      elements.push({ id: generateId(), type: 'or' })
      pos++
    } else {
      // Skip unknown characters
      pos++
    }
  }

  return elements
}

/**
 * Parse a bracket token [...] with optional modifiers
 */
function parseBracketToken(query: string, startPos: number): { element: BuilderElement; endPos: number } {
  // Find matching ]
  let bracketCount = 1
  let pos = startPos + 1
  while (pos < query.length && bracketCount > 0) {
    if (query[pos] === '[') bracketCount++
    else if (query[pos] === ']') bracketCount--
    pos++
  }

  const content = query.slice(startPos + 1, pos - 1).trim()
  let endPos = pos

  // Check for repetition modifier {n,m}, ?, or *
  let minCount: number | undefined
  let maxCount: number | undefined

  if (endPos < query.length && query[endPos] === '{') {
    const repEnd = query.indexOf('}', endPos)
    if (repEnd !== -1) {
      const repContent = query.slice(endPos + 1, repEnd)
      if (repContent.includes(',')) {
        const [minStr, maxStr] = repContent.split(',')
        minCount = parseInt(minStr.trim()) || 0
        maxCount = parseInt(maxStr.trim()) || 100
      } else {
        minCount = maxCount = parseInt(repContent.trim()) || 1
      }
      endPos = repEnd + 1
    }
  } else if (endPos < query.length && query[endPos] === '?') {
    minCount = 0
    maxCount = 1
    endPos++
  } else if (endPos < query.length && query[endPos] === '*') {
    minCount = 0
    maxCount = 100
    endPos++
  }

  // Empty brackets: [] or []{n,m} or []? or []*
  if (!content) {
    if (minCount !== undefined || maxCount !== undefined) {
      return {
        element: {
          id: generateId(),
          type: 'distance',
          minCount: minCount ?? 0,
          maxCount: maxCount ?? 1
        },
        endPos
      }
    }
    return {
      element: { id: generateId(), type: 'unspecified_token' },
      endPos
    }
  }

  // Has content: parse conditions
  const conditionGroups = parseConditions(content)

  return {
    element: {
      id: generateId(),
      type: 'normal_token',
      conditionGroups
    },
    endPos
  }
}

/**
 * Parse condition content inside brackets into ConditionGroups
 */
function parseConditions(content: string): ConditionGroup[] {
  // Check for | (OR) between conditions
  const orParts = splitByOperator(content, '|')

  if (orParts.length > 1) {
    // OR conditions - put all in one group with logic='or'
    const conditions: TokenCondition[] = []
    for (const part of orParts) {
      // Each OR part may contain AND conditions
      const andParts = splitByOperator(part.trim(), '&')
      for (const andPart of andParts) {
        const parsed = parseSingleCondition(andPart.trim())
        if (parsed) conditions.push(parsed)
      }
    }
    return [{ conditions, logic: 'or' }]
  }

  // AND conditions
  const andParts = splitByOperator(content, '&')
  const conditions: TokenCondition[] = []
  for (const part of andParts) {
    const parsed = parseSingleCondition(part.trim())
    if (parsed) conditions.push(parsed)
  }
  return [{ conditions, logic: 'and' }]
}

/**
 * Parse a single condition string like pos="NOUN" or word=="make"
 */
function parseSingleCondition(text: string): TokenCondition | null {
  if (!text) return null

  // Handle negation prefix
  let negated = false
  let cleanText = text
  if (cleanText.startsWith('!') && !cleanText.startsWith('!=') && !cleanText.startsWith('!==')) {
    negated = true
    cleanText = cleanText.slice(1).trim()
  }

  // Match: attribute operator "value"
  // Operators: ==, !==, !=, =
  const match = cleanText.match(/^(\w+)\s*(===?|!==?|=)\s*"([^"]*)"$/)
  if (!match) return null

  const [, attribute, operator, value] = match

  // Valid attributes
  const validAttributes = ['word', 'lemma', 'pos', 'tag', 'dep', 'headword', 'headlemma', 'headpos', 'headdep']
  if (!validAttributes.includes(attribute)) return null

  let finalOperator = operator as ComparisonOperator
  if (negated && operator === '=') {
    finalOperator = '!='
    negated = false
  } else if (negated && operator === '==') {
    finalOperator = '!=='
    negated = false
  }

  return {
    id: generateId(),
    attribute: attribute as TokenAttribute,
    operator: finalOperator,
    value
  }
}

/**
 * Split content by single-char operator, respecting quoted strings
 */
function splitByOperator(content: string, op: string): string[] {
  const parts: string[] = []
  let current: string[] = []
  let inQuotes = false

  for (const char of content) {
    if (char === '"') {
      inQuotes = !inQuotes
      current.push(char)
    } else if (char === op && !inQuotes) {
      parts.push(current.join(''))
      current = []
    } else {
      current.push(char)
    }
  }
  if (current.length) parts.push(current.join(''))
  return parts
}
